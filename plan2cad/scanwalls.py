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
