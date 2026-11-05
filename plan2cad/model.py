"""Structures de données géométriques partagées."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Poly:
    """Polyligne 2D en coordonnées monde (mm réels)."""
    pts: np.ndarray            # (N, 2) float64
    layer: str
    closed: bool = False
    width: float = 0.0         # épaisseur de trait mesurée (mm réels), 0 = inconnue
    disc: str = ""             # discipline : arc / elec / plumb / cvc / ""

    def length(self) -> float:
        p = self.pts
        if len(p) < 2:
            return 0.0
        d = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        if self.closed:
            d += np.linalg.norm(p[0] - p[-1])
        return float(d)


# --- transformations de similarité (rotation + échelle + translation) -------
# représentées par un couple de complexes (p, q) :  z' = p*z + q

Sim = tuple  # (p: complex, q: complex)
IDENTITY: Sim = (1 + 0j, 0 + 0j)


def sim_apply(sim: Sim, pts: np.ndarray) -> np.ndarray:
    p, q = sim
    z = pts[:, 0] + 1j * pts[:, 1]
    w = p * z + q
    return np.stack([w.real, w.imag], axis=1)


def sim_compose(first: Sim, then: Sim) -> Sim:
    """Applique `first` puis `then`."""
    p1, q1 = first
    p2, q2 = then
    return (p2 * p1, p2 * q1 + q2)


def sim_params(sim: Sim) -> dict:
    p, q = sim
    return {
        "scale": float(abs(p)),
        "rotation_deg": float(np.degrees(np.angle(p))),
        "tx": float(q.real),
        "ty": float(q.imag),
    }


def sim_from_pairs(src: np.ndarray, dst: np.ndarray) -> Sim:
    """Similarité qui envoie src[0]->dst[0] et src[1]->dst[1]."""
    s0 = complex(*src[0]); s1 = complex(*src[1])
    d0 = complex(*dst[0]); d1 = complex(*dst[1])
    p = (d1 - d0) / (s1 - s0)
    q = d0 - p * s0
    return (p, q)


def transform_polys(polys: list[Poly], sim: Sim) -> list[Poly]:
    out = []
    p = abs(sim[0])
    for pl in polys:
        out.append(Poly(sim_apply(sim, pl.pts), pl.layer, pl.closed,
                        pl.width * p, pl.disc))
    return out


def bbox(polys: list[Poly]):
    if not polys:
        return None
    mn = np.min([pl.pts.min(axis=0) for pl in polys], axis=0)
    mx = np.max([pl.pts.max(axis=0) for pl in polys], axis=0)
    return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


def sample_points(polys: list[Poly], step: float, max_pts: int = 60000) -> np.ndarray:
    """Échantillonne régulièrement les polylignes (pour ICP / métriques)."""
    out = []
    for pl in polys:
        p = pl.pts
        if pl.closed and len(p) > 2:
            p = np.vstack([p, p[:1]])
        if len(p) < 2:
            if len(p) == 1:
                out.append(p)
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        if total <= 0:
            out.append(p[:1])
            continue
        n = max(2, int(total / step) + 1)
        t = np.linspace(0, total, n)
        out.append(np.stack([np.interp(t, cum, p[:, 0]),
                             np.interp(t, cum, p[:, 1])], axis=1))
    if not out:
        return np.zeros((0, 2))
    pts = np.vstack(out)
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    return pts


@dataclass
class Text:
    """Texte lu sur le plan (monde, mm réels). pos = coin bas-gauche de la ligne de base."""
    pos: np.ndarray            # (2,)
    s: str
    height: float              # mm réels
    layer: str
    angle: float = 0.0         # degrés
    conf: float = 0.0


def transform_texts(texts: list, sim) -> list:
    out = []
    p = sim[0]
    for t in texts:
        out.append(Text(sim_apply(sim, np.asarray(t.pos, float)[None, :])[0], t.s, t.height * abs(p), t.layer,
                        t.angle + float(np.degrees(np.angle(p))), t.conf))
    return out
