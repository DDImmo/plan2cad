"""Extrusion 3D (OPTIONNELLE) : murs 2D -> volumes, réseaux posés à leur altitude.

Sorties : OBJ(+MTL), STL binaire, DXF 3DFACE, JSON (visionneuse web).
Z vers le haut, unités = mm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .dxf import write_dxf
from .layers import CHARTER, aci_hex, layer_color
from .model import Poly


@dataclass
class ExtrudeOptions:
    wall_height: float = 2700.0
    wall_thickness: float | None = None        # None = épaisseur mesurée, bornée
    min_thickness: float = 80.0
    max_thickness: float = 450.0
    default_thickness: float = 200.0
    elevations: dict = field(default_factory=lambda: {"elec": 2400.0, "plumb": 300.0, "cvc": 2500.0, "gaz": 400.0})
    sections: dict = field(default_factory=lambda: {"elec": 30.0, "plumb": 50.0, "cvc": 150.0, "gaz": 40.0})


def _boxes(pts: np.ndarray, closed: bool, t: float, z0: float, z1: float) -> np.ndarray:
    """Un prisme par segment (prolongé de t/2 aux extrémités pour combler les coudes)."""
    p = pts
    if closed and len(p) > 2:
        p = np.vstack([p, p[:1]])
    a, b = p[:-1], p[1:]
    d = b - a
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    a, b, d, L = a[ok], b[ok], d[ok], L[ok]
    if len(a) == 0:
        return np.zeros((0, 3, 3))
    u = d / L[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    h = t / 2
    a = a - u * h
    b = b + u * h
    c = np.stack([a + nrm * h, a - nrm * h, b - nrm * h, b + nrm * h], 1)       # (S,4,2)
    S = len(a)
    bot = np.concatenate([c, np.full((S, 4, 1), z0)], 2)
    top = np.concatenate([c, np.full((S, 4, 1), z1)], 2)
    V = np.concatenate([bot, top], 1)                                           # (S,8,3)
    tri_idx = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for i in range(4):
        j = (i + 1) % 4
        tri_idx += [(i, j, j + 4), (i, j + 4, i + 4)]
    ti = np.array(tri_idx)
    return V[:, ti, :].reshape(-1, 3, 3)


DOOR_H, SILL_H = 2100.0, 900.0          # hauteur de porte / d'allège de fenêtre (mm)


def build_mesh(polys: list[Poly], opts: ExtrudeOptions, z_base: float = 0.0,
               height: float | None = None) -> dict[str, np.ndarray]:
    """Renvoie {calque: triangles (N,3,3)}."""
    out: dict[str, list] = {}
    wall_h = height or opts.wall_height
    for pl in polys:
        if len(pl.pts) < 2:
            continue
        if pl.layer in ("ARC-PORTE", "ARC-FENETRE", "ARC-ESCALIER"):
            # symboles 2D (battants, arcs, marches) : pas de volume. Seuls les repères d'ouverture (trait épais = épaisseur
            # du mur, posé en travers de l'ouverture) donnent un panneau + linteau (+ allège pour une fenêtre).
            if pl.layer != "ARC-ESCALIER" and pl.width >= 80 and len(pl.pts) == 2 and not pl.closed:
                dh = min(DOOR_H, 0.78 * wall_h)
                parts = []
                if pl.layer == "ARC-PORTE":
                    parts += [(40.0, 0.0, dh), (pl.width, dh, wall_h)]
                else:
                    sill = min(SILL_H, 0.34 * wall_h)
                    parts += [(pl.width, 0.0, sill), (30.0, sill, dh), (pl.width, dh, wall_h)]
                for th, za, zb in parts:
                    tri = _boxes(pl.pts, False, th, z_base + za, z_base + zb)
                    if len(tri):
                        out.setdefault(pl.layer, []).append(tri)
            continue
        if pl.disc == "arc" or pl.layer.startswith("ARC"):
            if opts.wall_thickness:
                t = opts.wall_thickness
            else:
                t = pl.width if pl.width > 0 else opts.default_thickness
                t = float(np.clip(t, opts.min_thickness, opts.max_thickness))
            tri = _boxes(pl.pts, pl.closed, t, z_base, z_base + wall_h)
        elif pl.layer.startswith("Z-"):
            continue
        else:
            d = pl.disc or pl.layer.split("-")[0].lower()
            s = opts.sections.get(d, 40.0)
            z = z_base + opts.elevations.get(d, wall_h * 0.9)
            tri = _boxes(pl.pts, pl.closed, s, z - s / 2, z + s / 2)
        if len(tri):
            out.setdefault(pl.layer, []).append(tri)
    return {k: np.concatenate(v) for k, v in out.items()}


def merge_meshes(meshes: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for m in meshes:
        for k, v in m.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ exports
def export_obj(path, mesh: dict[str, np.ndarray]):
    mtl = str(path)[:-4] + ".mtl"
    with open(mtl, "w") as fm:
        for name in mesh:
            col = aci_hex(layer_color(name))
            r, g, b = (int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))
            fm.write(f"newmtl {name}\nKd {r:.3f} {g:.3f} {b:.3f}\n\n")
    base = 1
    with open(path, "w") as fo:
        fo.write(f"mtllib {mtl.split('/')[-1]}\n")
        for name, tri in mesh.items():
            fo.write(f"g {name}\nusemtl {name}\n")
            v = tri.reshape(-1, 3)
            fo.write("".join(f"v {x:.2f} {y:.2f} {z:.2f}\n" for x, y, z in v))
            n = len(tri)
            f = np.arange(n * 3).reshape(n, 3) + base
            fo.write("".join(f"f {a} {b} {c}\n" for a, b, c in f))
            base += n * 3


def export_stl(path, mesh: dict[str, np.ndarray]):
    tri = np.concatenate(list(mesh.values())) if mesh else np.zeros((0, 3, 3))
    n = len(tri)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    nrm = np.cross(e1, e2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    arr = np.zeros(n, dt)
    arr["n"] = nrm
    arr["v"] = tri
    with open(path, "wb") as f:
        f.write(b"plan2cad".ljust(80, b" "))
        f.write(np.uint32(n).tobytes())
        f.write(arr.tobytes())


def export_dxf3d(path, mesh: dict[str, np.ndarray]):
    ents = []
    for name, tri in mesh.items():
        for t in tri:
            ents.append(("face", name, t))
    write_dxf(path, ents, {k: layer_color(k) for k in mesh})


def mesh_to_json(mesh: dict[str, np.ndarray]) -> dict:
    layers = []
    for name, tri in mesh.items():
        layers.append(dict(name=name, color=aci_hex(layer_color(name)),
                           tris=len(tri),
                           pos=np.round(tri.reshape(-1), 0).astype(int).tolist()))
    allv = np.concatenate([t.reshape(-1, 3) for t in mesh.values()]) if mesh else np.zeros((1, 3))
    return dict(layers=layers, min=allv.min(0).tolist(), max=allv.max(0).tolist(), unit="mm")
