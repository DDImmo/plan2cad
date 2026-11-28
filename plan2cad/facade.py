"""Fenêtres et baies vitrées d'un plan vectoriel : les vitrages de façade sont tracés comme de fins rectangles
cloisonnés collés au mur extérieur (dans l'épaisseur du mur), à ne pas confondre avec le mobilier posé contre la façade.

Critères : bande allongée (≥ 0,6 m de long, ≤ 0,7 m de large, rapport ≥ 1,8), parallèle au mur extérieur le plus proche
(≤ 12°) et dans son épaisseur (≤ 25 cm de l'axe tracé)."""
from __future__ import annotations

import cv2
import numpy as np

SRC_LAYERS = ("ARC-MOBILIER", "ARC-DETAIL")
WINDOW_LAYER = "ARC-FENETRE"


def _dense(pts, step):
    out, ang = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        if L < 1e-6:
            continue
        n = max(1, int(L / step))
        t = np.arange(n + 1) / n
        out.append(a[None, :] + (b - a)[None, :] * t[:, None])
        ang += [np.degrees(np.arctan2(*(b - a)[::-1])) % 180.0] * (n + 1)
    return (np.vstack(out), np.array(ang)) if out else (np.zeros((0, 2)), np.zeros(0))


def tag_windows(polys, max_dist: float = 250.0, min_long: float = 600.0, max_short: float = 700.0,
                min_aspect: float = 1.8, ang_tol: float = 12.0) -> int:
    """Passe en ARC-FENETRE les bandes vitrées collées à la façade. Renvoie leur nombre."""
    from scipy.spatial import cKDTree
    P, A = [], []
    for p in polys:
        if p.layer == "ARC-MURS-EXT" and len(p.pts) >= 2:
            d, a = _dense(np.asarray(p.pts, float), 100.0)
            P.append(d); A.append(a)
    if not P:
        return 0
    P, A = np.vstack(P), np.concatenate(A)
    if len(P) < 20:
        return 0
    tree = cKDTree(P)
    n = 0
    for p in polys:
        if p.layer not in SRC_LAYERS or len(p.pts) < 2:
            continue
        pts = np.asarray(p.pts, np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        long_, short = max(w, h), min(w, h)
        if long_ < min_long or short > max_short or long_ < min_aspect * max(short, 1.0):
            continue
        d, a = _dense(np.asarray(p.pts, float), 150.0)
        if len(d) == 0:
            continue
        dist, idx = tree.query(d)
        # segments LONGS du candidat (les traverses des croisillons n'entrent pas dans le critère de direction)
        seg = np.diff(np.asarray(p.pts, float), axis=0)
        sl = np.hypot(seg[:, 0], seg[:, 1])
        if not len(sl):
            continue
        k = int(np.argmax(sl))
        ang = np.degrees(np.arctan2(seg[k, 1], seg[k, 0])) % 180.0
        da = abs(ang - A[tree.query(np.asarray(p.pts, float)[[k]])[1][0]])
        da = min(da, 180.0 - da)
        if float(np.median(dist)) <= max_dist and da <= ang_tol:
            p.layer = WINDOW_LAYER
            n += 1
    return n
