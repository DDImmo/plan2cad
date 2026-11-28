"""Trame des murs d'un SCAN : longs traits droits (horizontaux / verticaux) de l'image, au centre des murs à double trait.

La vectorisation par squelette fragmente les murs d'un scan (traits irréguliers, bouts manquants) : les pièces ne se
ferment pas et les métrés en trouvent trop peu. Les longs traits droits de l'image, eux, sont continus. On les ajoute
comme calque à part (ARC-MURS-TRAME) ; ils servent aux métrés et se masquent d'un clic dans le logiciel de CAO."""
from __future__ import annotations

import cv2
import numpy as np

LAYER = "ARC-MURS-TRAME"


def long_wall_segments(ink: np.ndarray, mm_px: float, min_mm: float = 6.0, pair_mm: float = 4.5):
    """Segments (x0, y0, x1, y1) en px : un par trait long. Les deux traits d'un mur à double trait (écart ≤ pair_mm)
    sont fondus en un seul segment central."""
    L = max(15, int(round(min_mm / mm_px)))
    g = max(3, int(round(pair_mm / mm_px)) | 1)
    segs = []
    for horiz in (True, False):
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if horiz else (1, L))
        m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker)
        fuse = cv2.getStructuringElement(cv2.MORPH_RECT, (1, g) if horiz else (g, 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, fuse)                 # fond les deux traits d'un double trait
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h = st[i, :4]
            if horiz and w >= L and w >= 2 * h:
                segs.append((x, y + h / 2.0, x + w, y + h / 2.0))
            elif not horiz and h >= L and h >= 2 * w:
                segs.append((x + w / 2.0, y, x + w / 2.0, y + h))
    return segs


def extend_to_neighbours(segs, max_px: float, tol_px: float = 6.0):
    """Recrée un bord absent : un bout de mur libre (aucun autre mur ne le touche) est prolongé, dans sa direction, jusqu'au
    premier mur perpendiculaire rencontré à moins de `max_px`. Ne relie pas deux murs alignés (ça fermerait les portes)."""
    segs = [list(map(float, s)) for s in segs]
    H = [s for s in segs if abs(s[1] - s[3]) < 1e-6]
    V = [s for s in segs if abs(s[0] - s[2]) < 1e-6]

    def touched(px, py, own):
        for s in segs:
            if s is own:
                continue
            if min(s[0], s[2]) - tol_px <= px <= max(s[0], s[2]) + tol_px and min(s[1], s[3]) - tol_px <= py <= max(s[1], s[3]) + tol_px:
                return True
        return False

    for s in H:
        y = s[1]
        for end, sign in ((0, -1), (2, 1)):
            x = s[end]
            if touched(x, y, s):
                continue
            best = None
            for v in V:
                vx = v[0]
                if min(v[1], v[3]) - tol_px <= y <= max(v[1], v[3]) + tol_px and 0 < sign * (vx - x) <= max_px:
                    if best is None or abs(vx - x) < abs(best - x):
                        best = vx
            if best is not None:
                s[end] = best
    for s in V:
        x = s[0]
        for end, sign in ((1, -1), (3, 1)):
            y = s[end]
            if touched(x, y, s):
                continue
            best = None
            for h in H:
                hy = h[1]
                if min(h[0], h[2]) - tol_px <= x <= max(h[0], h[2]) + tol_px and 0 < sign * (hy - y) <= max_px:
                    if best is None or abs(hy - y) < abs(best - y):
                        best = hy
            if best is not None:
                s[end] = best
    return [tuple(s) for s in segs]


def snap_walls(polys, segs_mm, tol_mm: float):
    """Redresse les murs tremblants du scan : chaque sommet d'un mur proche (≤ tol) d'un segment de la trame est ramené
    sur ce segment ; les sommets alignés qui en résultent sont supprimés. Renvoie le nombre de murs modifiés."""
    import numpy as np
    S = np.asarray(segs_mm, float)
    if not len(S):
        return 0
    horiz = np.abs(S[:, 1] - S[:, 3]) < 1e-6
    n = 0
    for p in polys:
        if p.layer not in ("ARC-MURS-INT", "ARC-MURS-EXT") or len(p.pts) < 2:
            continue
        pts = np.asarray(p.pts, float).copy()
        moved = False
        for i, (x, y) in enumerate(pts):
            best, bd = None, tol_mm
            for s, hz in zip(S, horiz):
                if hz:
                    if min(s[0], s[2]) - tol_mm <= x <= max(s[0], s[2]) + tol_mm and abs(y - s[1]) < bd:
                        best, bd = (x, s[1]), abs(y - s[1])
                else:
                    if min(s[1], s[3]) - tol_mm <= y <= max(s[1], s[3]) + tol_mm and abs(x - s[0]) < bd:
                        best, bd = (s[0], y), abs(x - s[0])
            if best is not None:
                pts[i] = best
                moved = True
        if not moved:
            continue
        keep = [0]
        for i in range(1, len(pts) - 1):
            a, b, c = pts[keep[-1]], pts[i], pts[i + 1]
            if np.hypot(*(b - a)) < 1e-6:
                continue
            cr = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cr > 1e-6 * max(1.0, np.hypot(*(c - a)) ** 2) and cr / max(np.hypot(*(c - a)), 1e-9) > 1.0:
                keep.append(i)
        keep.append(len(pts) - 1)
        newp = pts[keep]
        if len(newp) >= 2:
            p.pts = newp
            n += 1
    return n
