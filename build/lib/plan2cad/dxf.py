"""Lecteur / écrivain DXF minimal, sans dépendance (ASCII, format R12 = AC1009).

Le R12 est lu par FreeCAD, QCAD, AutoCAD, LibreCAD, BricsCAD, etc.
Entités écrites : POLYLINE, LINE, POINT, TEXT, 3DFACE.
Entités lues : LINE, POLYLINE/VERTEX, LWPOLYLINE, POINT, CIRCLE, ARC.
"""
from __future__ import annotations

import math
import numpy as np


def _f(v) -> str:
    return f"{float(v):.4f}"


def write_dxf(path, entities, layer_colors: dict[str, int]):
    """entities : liste de tuples
        ("polyline", layer, pts(N,2|3), closed)
        ("line", layer, p0, p1)
        ("point", layer, p)
        ("face", layer, pts(3|4,3))
    """
    layers = dict(layer_colors)
    for e in entities:
        layers.setdefault(e[1], 7)
    o: list[str] = []

    def g(code, val):
        o.append(str(code))
        o.append(str(val))

    g(0, "SECTION"); g(2, "HEADER")
    g(9, "$ACADVER"); g(1, "AC1009")
    g(9, "$INSUNITS"); g(70, 4)          # millimètres
    g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "TABLES")
    g(0, "TABLE"); g(2, "LTYPE"); g(70, 1)
    g(0, "LTYPE"); g(2, "CONTINUOUS"); g(70, 0); g(3, "Solid line"); g(72, 65); g(73, 0); g(40, "0.0")
    g(0, "ENDTAB")
    g(0, "TABLE"); g(2, "LAYER"); g(70, len(layers))
    for name, col in layers.items():
        g(0, "LAYER"); g(2, name); g(70, 0); g(62, int(col)); g(6, "CONTINUOUS")
    g(0, "ENDTAB")
    g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "BLOCKS"); g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "ENTITIES")
    for e in entities:
        kind, layer = e[0], e[1]
        if kind == "polyline":
            pts = np.asarray(e[2], dtype=float)
            closed = bool(e[3])
            g(0, "POLYLINE"); g(8, layer); g(66, 1)
            g(10, "0.0"); g(20, "0.0"); g(30, "0.0"); g(70, 1 if closed else 0)
            for p in pts:
                g(0, "VERTEX"); g(8, layer)
                g(10, _f(p[0])); g(20, _f(p[1])); g(30, _f(p[2]) if len(p) > 2 else "0.0")
            g(0, "SEQEND"); g(8, layer)
        elif kind == "line":
            p0, p1 = e[2], e[3]
            g(0, "LINE"); g(8, layer)
            g(10, _f(p0[0])); g(20, _f(p0[1])); g(30, _f(p0[2]) if len(p0) > 2 else "0.0")
            g(11, _f(p1[0])); g(21, _f(p1[1])); g(31, _f(p1[2]) if len(p1) > 2 else "0.0")
        elif kind == "point":
            p = e[2]
            g(0, "POINT"); g(8, layer)
            g(10, _f(p[0])); g(20, _f(p[1])); g(30, _f(p[2]) if len(p) > 2 else "0.0")
        elif kind == "text":
            # ("text", layer, pos, hauteur, chaîne, angle_deg)
            p, hgt, txt = e[2], float(e[3]), str(e[4]).replace("\n", " ")
            txt = txt.encode("cp1252", errors="replace").decode("cp1252")
            g(0, "TEXT"); g(8, layer)
            g(10, _f(p[0])); g(20, _f(p[1])); g(30, "0.0")
            g(40, _f(hgt)); g(1, txt)
            if len(e) > 5 and abs(float(e[5])) > 1e-6:
                g(50, _f(e[5]))
        elif kind == "face":
            pts = np.asarray(e[2], dtype=float)
            if len(pts) == 3:
                pts = np.vstack([pts, pts[2]])
            g(0, "3DFACE"); g(8, layer)
            for i, p in enumerate(pts):
                g(10 + i, _f(p[0])); g(20 + i, _f(p[1])); g(30 + i, _f(p[2]))
    g(0, "ENDSEC")
    g(0, "EOF")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(o) + "\n")


# ------------------------------------------------------------------ lecture
def read_dxf(path):
    """Renvoie (entities, layer_colors, skipped).
    entities : liste de dicts {type:'poly'|'point', layer, pts(N,2), closed}"""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"AutoCAD Binary DXF"):
        raise ValueError("DXF binaire non supporté : réexporter en DXF ASCII.")
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("cp1252", errors="replace")
    lines = txt.splitlines()
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        try:
            pairs.append((int(lines[i].strip()), lines[i + 1].strip()))
        except ValueError:
            continue

    layer_colors: dict[str, int] = {}
    ents: list[dict] = []
    skipped: dict[str, int] = {}

    # sections
    i, n = 0, len(pairs)
    section = None
    cur_table = None
    # --- tables (couleurs des calques)
    j = 0
    while j < n:
        c, v = pairs[j]
        if c == 0 and v == "SECTION" and j + 1 < n:
            section = pairs[j + 1][1]
        elif c == 0 and v == "ENDSEC":
            section = None
        elif section == "TABLES" and c == 0 and v == "LAYER":
            name, col = None, 7
            k = j + 1
            while k < n and pairs[k][0] != 0:
                if pairs[k][0] == 2:
                    name = pairs[k][1]
                elif pairs[k][0] == 62:
                    col = abs(int(float(pairs[k][1])))
                k += 1
            if name:
                layer_colors[name] = col
        j += 1

    # --- entités
    def collect(start):
        k = start + 1
        d = []
        while k < n and pairs[k][0] != 0:
            d.append(pairs[k]); k += 1
        return d, k

    j = 0
    in_ent = False
    while j < n:
        c, v = pairs[j]
        if c == 0 and v == "SECTION" and j + 1 < n and pairs[j + 1][1] == "ENTITIES":
            in_ent = True; j += 2; continue
        if c == 0 and v == "ENDSEC":
            in_ent = False
        if not in_ent or c != 0:
            j += 1; continue
        d, k = collect(j)
        get = lambda code, dflt=None: next((x for cc, x in d if cc == code), dflt)
        layer = get(8, "0")
        if v == "LINE":
            p = np.array([[float(get(10, 0)), float(get(20, 0))],
                          [float(get(11, 0)), float(get(21, 0))]])
            ents.append(dict(type="poly", layer=layer, pts=p, closed=False))
        elif v == "LWPOLYLINE":
            xs = [float(x) for cc, x in d if cc == 10]
            ys = [float(x) for cc, x in d if cc == 20]
            flag = int(float(get(70, 0)))
            if len(xs) >= 2:
                ents.append(dict(type="poly", layer=layer,
                                 pts=np.stack([xs, ys], 1), closed=bool(flag & 1)))
        elif v == "POLYLINE":
            flag = int(float(get(70, 0)))
            verts = []
            k2 = k
            while k2 < n:
                cc, vv = pairs[k2]
                if cc == 0 and vv == "VERTEX":
                    dd, k3 = collect(k2)
                    gx = next((x for c2, x in dd if c2 == 10), "0")
                    gy = next((x for c2, x in dd if c2 == 20), "0")
                    verts.append([float(gx), float(gy)])
                    k2 = k3
                elif cc == 0 and vv == "SEQEND":
                    _, k2 = collect(k2)
                    break
                else:
                    break
            if len(verts) >= 2:
                ents.append(dict(type="poly", layer=layer, pts=np.array(verts),
                                 closed=bool(flag & 1)))
            k = k2
        elif v == "POINT":
            ents.append(dict(type="point", layer=layer,
                             pts=np.array([[float(get(10, 0)), float(get(20, 0))]]),
                             closed=False))
        elif v in ("CIRCLE", "ARC"):
            cx, cy, r = float(get(10, 0)), float(get(20, 0)), float(get(40, 0))
            a0, a1 = (0.0, 360.0) if v == "CIRCLE" else (float(get(50, 0)), float(get(51, 360)))
            if a1 <= a0:
                a1 += 360
            m = max(8, int((a1 - a0) / 7.5))
            t = np.radians(np.linspace(a0, a1, m + 1))
            p = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)
            ents.append(dict(type="poly", layer=layer, pts=p, closed=(v == "CIRCLE")))
        else:
            skipped[v] = skipped.get(v, 0) + 1
        j = k
    return ents, layer_colors, skipped
