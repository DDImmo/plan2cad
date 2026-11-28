"""Portes à battant sur un scan, SANS connaître l'échelle : appariement de gabarits « vantail + quart de cercle ».

Un symbole de porte est un segment (le vantail, long comme la porte) + un quart de cercle de même rayon qui part de son
extrémité. On corrèle ce gabarit (8 orientations, rayons étagés) avec l'encre : le vantail et l'arc doivent être encrés,
et le quart de disque balayé doit être VIDE. Les arcs de chaises, de vasques et de mobilier n'ont ni vantail, ni
intérieur vide : ils sont écartés. Le rayon trouvé EST la largeur de la porte (en mm papier), donc l'échelle :
une porte simple fait environ 0,80-0,90 m.
"""
from __future__ import annotations

import cv2
import numpy as np

DOOR_NOMINAL_MM = 750.0     # le vantail mesuré fait 0,65 à 0,85 m selon le plan : milieu de fourchette, ordre de grandeur seulement


def _templates(r: int):
    """8 gabarits (vantail le long de ±x / ±y, arc d'un côté ou de l'autre) pour le rayon r (px).
    Chaque gabarit : (arc, vantail, intérieur) en float32, charnière en (r, r) du gabarit de taille (2r+1)."""
    n = 2 * r + 5
    c = r + 2
    out = []
    for lx, ly in ((1, 0), (-1, 0), (0, 1), (0, -1)):            # direction du vantail
        for side in (1, -1):                                      # de quel côté tourne l'arc
            arc = np.zeros((n, n), np.uint8)
            leaf = np.zeros((n, n), np.uint8)
            inner = np.zeros((n, n), np.uint8)
            tip = (c + lx * r, c + ly * r)
            cv2.line(leaf, (c, c), tip, 1, 1)
            # l'arc va de l'extrémité du vantail à la position « fermée » (vantail tourné de 90°)
            a0 = np.degrees(np.arctan2(ly, lx))
            a1 = a0 + 90.0 * side
            cv2.ellipse(arc, (c, c), (r, r), 0, min(a0, a1), max(a0, a1), 1, 1)
            ang = np.radians(np.linspace(a0, a1, 40)[4:-4])
            for rr in np.linspace(0.25 * r, 0.8 * r, 6):
                for t in ang:
                    inner[int(round(c + rr * np.sin(t))), int(round(c + rr * np.cos(t)))] = 1
            ring_i = np.zeros((n, n), np.uint8)
            ring_o = np.zeros((n, n), np.uint8)
            if r > 7:
                cv2.ellipse(ring_i, (c, c), (r - 4, r - 4), 0, min(a0, a1) + 8, max(a0, a1) - 8, 1, 1)
            cv2.ellipse(ring_o, (c, c), (r + 4, r + 4), 0, min(a0, a1) + 8, max(a0, a1) - 8, 1, 1)
            inner = cv2.dilate(inner, np.ones((3, 3), np.uint8))
            inner[cv2.dilate(leaf, np.ones((5, 5), np.uint8)) > 0] = 0
            out.append(((lx, ly, side), arc.astype(np.float32), leaf.astype(np.float32), inner.astype(np.float32), c,
                        ring_i.astype(np.float32), ring_o.astype(np.float32)))
    return out


def detect_swing_doors(ink: np.ndarray, mm_px: float, rmin_mm: float = 1.6, rmax_mm: float = 9.0,
                       work_mm: float = 0.17, min_arc: float = 0.88, min_leaf: float = 0.9, max_inner: float = 0.10,
                       min_wall: float = 0.8, max_ring: float = 0.3):
    """Renvoie une liste de dict(cx, cy (charnière, px image), r_px, r_mm (papier), lx, ly, side, score).
    `ink` : uint8 255 = encre (image pleine résolution)."""
    ds = max(1, int(round(work_mm / mm_px)))
    img = ink if ds == 1 else cv2.resize(ink, (ink.shape[1] // ds, ink.shape[0] // ds), interpolation=cv2.INTER_AREA)
    img = (img > (0 if ds == 1 else 60)).astype(np.uint8)
    d = cv2.dilate(img, np.ones((3, 3), np.uint8)).astype(np.float32)
    d2 = cv2.dilate(img, np.ones((9, 9), np.uint8)).astype(np.float32)     # tolérance transversale pour le mur
    mp = mm_px * ds
    rs = []
    r = max(6, int(round(rmin_mm / mp)))
    while r <= rmax_mm / mp:
        rs.append(r)
        r = max(r + 1, int(round(r * 1.06)))
    found = []
    for r in rs:
        for (lx, ly, side), arc, leaf, inner, c, ring_i, ring_o in _templates(r):
            na, nl, ni = float(arc.sum()), float(leaf.sum()), float(inner.sum())
            if na < 8 or nl < 4:
                continue
            ca = cv2.matchTemplate(d, arc, cv2.TM_CCORR) / na
            cl = cv2.matchTemplate(d, leaf, cv2.TM_CCORR) / nl
            ok = (ca >= min_arc) & (cl >= min_leaf)
            if not ok.any():
                continue
            ci = cv2.matchTemplate(d, inner, cv2.TM_CCORR) / max(ni, 1.0)
            ok &= ci <= max_inner
            for rk in (ring_i, ring_o):
                nr = float(rk.sum())
                if nr >= 6 and ok.any():
                    ok &= (cv2.matchTemplate(d, rk, cv2.TM_CCORR) / nr) <= max_ring
            ys, xs = np.nonzero(ok)
            if len(xs):
                # le mur : une porte s'ouvre DANS un mur, qui se poursuit de part et d'autre de l'ouverture (direction du
                # vantail fermé = fin de l'arc) ; ce test écarte les arcs de mobilier / de cartouche sans mur
                a1 = np.radians(np.degrees(np.arctan2(ly, lx)) + 90.0 * side)
                ux, uy = np.cos(a1), np.sin(a1)
                cx0, cy0 = xs + c, ys + c
                tA = np.linspace(-0.7 * r, -3, 14)
                tB = np.linspace(r + 3, 1.7 * r, 14)
                def cover(ts):
                    px = np.clip(np.round(cx0[:, None] + ts[None, :] * ux).astype(int), 0, d2.shape[1] - 1)
                    py = np.clip(np.round(cy0[:, None] + ts[None, :] * uy).astype(int), 0, d2.shape[0] - 1)
                    return d2[py, px].mean(axis=1)
                wa, wb = cover(tA), cover(tB)
                good = (wa >= min_wall) & (wb >= min_wall)
                ys, xs = ys[good], xs[good]
            for y, x in zip(ys, xs):
                found.append(dict(cx=(x + c) * ds, cy=(y + c) * ds, r_px=r * ds, r_mm=r * mp, lx=lx, ly=ly, side=side,
                                  score=float(ca[y, x] + cl[y, x] - ci[y, x])))
    # suppression des non-maxima : une porte = une détection (la meilleure, à ~0,8 r près)
    found.sort(key=lambda f: -f["score"] - 0.002 * f["r_px"])
    keep = []
    for f in found:
        if all(np.hypot(f["cx"] - g["cx"], f["cy"] - g["cy"]) > 0.9 * max(f["r_px"], g["r_px"]) for g in keep):
            keep.append(f)
    return keep


def mark_doubles(doors):
    """Deux battants face à face (portes doubles) : chacun ne fait que la moitié de la baie ; ils faussent l'échelle."""
    for d in doors:
        d["double"] = False
    for i, a in enumerate(doors):
        for b in doors[i + 1:]:
            r = max(a["r_px"], b["r_px"])
            if abs(a["r_px"] - b["r_px"]) <= 0.2 * r and 1.5 * r <= np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) <= 2.6 * r:
                a["double"] = b["double"] = True
    return doors


def scale_from_doors(doors, door_mm: float = DOOR_NOMINAL_MM, min_n: int = 3):
    """Échelle d'après les rayons des portes SIMPLES (mm papier). Classe de rayon dominante ±10 %, au moins `min_n` portes.
    Approximative : un battant fait 0,70 à 0,95 m selon le plan ; à confirmer par une cote ou --scale."""
    from .dimensions import STD_SCALES
    mark_doubles(doors)
    rs = np.sort(np.asarray([d["r_mm"] for d in doors if not d["double"]], float))
    if len(rs) < min_n:
        return None
    best = (0, 0.0)
    for r0 in rs:
        grp = rs[(rs >= r0 / 1.10) & (rs <= r0 * 1.10)]
        if len(grp) > best[0]:
            best = (len(grp), float(np.median(grp)))
    n, med = best
    if n < min_n or n < 0.4 * len(rs):
        return None
    s = door_mm / med
    std = min(STD_SCALES, key=lambda x: abs(np.log(s / x)))
    if abs(s / std - 1) <= 0.04:
        s = float(std)
    return dict(scale=round(float(s), 1), n=int(n), of=len(rs), largeur_mm_papier=round(med, 2),
                source="largeur des portes (battants ≈ 0,75 m ±0,1) : ordre de grandeur", approx=True)


def to_openings(doors, wt: float):
    """Portes à battant -> ouvertures au format de openings.detect_openings (rect, hinge, sx, sy : même convention)."""
    out = []
    for d in doors:
        lx, ly, side = d["lx"], d["ly"], d["side"]
        a1 = np.radians(np.degrees(np.arctan2(ly, lx)) + 90.0 * side)
        ux, uy = int(round(np.cos(a1))), int(round(np.sin(a1)))     # direction du vantail fermé = axe de l'ouverture
        hx, hy, r = d["cx"], d["cy"], d["r_px"]
        h = 0.5 * wt
        if ux != 0:
            rect = (min(hx, hx + ux * r), hy - h, max(hx, hx + ux * r), hy + h)
            orient = "h"
        else:
            rect = (hx - h, min(hy, hy + uy * r), hx + h, max(hy, hy + uy * r))
            orient = "v"
        out.append(dict(kind="porte", conf=0.9, orient=orient, rect=tuple(int(round(v)) for v in rect),
                        hinge=(hx, hy), sx=int(ux + lx), sy=int(uy + ly), cut=False, swing=True,
                        r_mm_paper=round(d["r_mm"], 2), why="vantail + arc de battant"))
    return out


def merge_openings(openings, swing_openings, mm_px: float, scale: float):
    """Les portes trouvées par gabarit remplacent les portes / ouvertures du détecteur de murs qui les recouvrent."""
    keep = []
    for o in openings:
        if o["kind"] in ("porte", "ouverture"):
            cx, cy = (o["rect"][0] + o["rect"][2]) / 2, (o["rect"][1] + o["rect"][3]) / 2
            if any(np.hypot(cx - (s["rect"][0] + s["rect"][2]) / 2, cy - (s["rect"][1] + s["rect"][3]) / 2)
                   < 0.8 * max(s["rect"][2] - s["rect"][0], s["rect"][3] - s["rect"][1]) for s in swing_openings):
                continue
        keep.append(o)
    for s in swing_openings:
        w = max(s["rect"][2] - s["rect"][0], s["rect"][3] - s["rect"][1])
        s["width_mm"] = round(w * mm_px * scale)
    return keep + swing_openings
