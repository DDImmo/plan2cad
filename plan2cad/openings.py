"""Portes, fenêtres, escaliers : détectés sur le scan (pixels) puis utilisés pour reclasser les tracés.

Principe (plans à murs pleins, ou à double trait déjà comblé) :
  * épaisseur des murs = épaisseur dominante des traits épais ;
  * un « mur » = ce qui reste de l'encre après ouverture morphologique (les traits fins disparaissent) ;
  * deux tronçons de mur alignés, séparés de 0,5 à 3 m, délimitent une ouverture ;
  * dans l'ouverture : arc de battant à côté -> porte ; traits fins en travers -> fenêtre ; sinon baie ouverte ;
  * escalier = série de ≥ 4 traits parallèles de même longueur, espacés comme des marches (17-42 cm).
Chaque détection porte une confiance. Rien n'est supprimé : les tracés sont seulement reclassés par calque.
"""
from __future__ import annotations

import cv2
import numpy as np

from .layers import split_walls_by_width

DOOR_LAYER, WINDOW_LAYER, STAIR_LAYER = "ARC-PORTE", "ARC-FENETRE", "ARC-ESCALIER"


def wall_thickness(ink: np.ndarray, mm_px: float):
    """Épaisseur (px) des murs pleins, ou None s'il n'y en a pas."""
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((5, 5), np.uint8)) - 1e-3) & (dt > 1.5)
    w = 2 * dt[ridge]
    if len(w) < 50:
        return None
    thr = split_walls_by_width(w * mm_px)
    thick = w[w > (thr / mm_px if thr else 0)]
    wt = float(np.median(thick)) if len(thick) else float(np.median(w))
    return wt if wt * mm_px >= 1.0 else None


def _segments(wall: np.ndarray, wt: float):
    """Tronçons horizontaux de mur : (x0, x1, y0, y1)."""
    L = max(5, int(2.0 * wt))
    hw = cv2.morphologyEx(wall, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    n, _lab, st, _ = cv2.connectedComponentsWithStats(hw, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w >= 2 * wt and h <= 1.8 * wt and a >= 0.5 * w * h:
            out.append((int(x), int(x + w), int(y), int(y + h)))
    return out


def _gaps(wall: np.ndarray, wt: float, gmin: float, gmax: float):
    segs = sorted(_segments(wall, wt), key=lambda s: s[0])
    out = []
    for i, a in enumerate(segs):
        best = None
        for b in segs:
            if b is a or b[0] < a[1] - 1:
                continue
            gap = b[0] - a[1]
            if gap < gmin or gap > gmax:
                continue
            if abs(a[2] - b[2]) > 0.3 * wt or abs(a[3] - b[3]) > 0.3 * wt:
                continue
            if best is None or gap < best[0] - a[1]:
                best = b
        if best is None:
            continue
        x0, x1 = a[1], best[0]
        y0, y1 = max(a[2], best[2]), min(a[3], best[3])
        if y1 - y0 < 0.6 * wt:
            continue
        cover = float((wall[y0:y1, x0:x1] > 0).mean())
        if cover > 0.2:                                    # un mur traverse : pas une ouverture
            continue
        out.append((x0, y0, x1, y1))
    return out


def _arc_score(ink_d: np.ndarray, hx: float, hy: float, r: float, sx: int, sy: int) -> float:
    """Part des angles 5°-85° où l'on trouve de l'encre à la distance r (±12 %) du charnière, dans le quadrant (sx, sy)."""
    H, W = ink_d.shape
    hit = tot = 0
    for ang in np.radians(np.linspace(6, 84, 28)):
        tot += 1
        for rr in np.linspace(0.88 * r, 1.12 * r, 7):
            x = int(round(hx + sx * rr * np.cos(ang)))
            y = int(round(hy + sy * rr * np.sin(ang)))
            if 0 <= x < W and 0 <= y < H and ink_d[y, x]:
                hit += 1
                break
    return hit / max(1, tot)


def _classify(ink: np.ndarray, ink_d: np.ndarray, rect, wt: float):
    x0, y0, x1, y1 = rect
    g = x1 - x0
    yc = (y0 + y1) / 2
    best = (0.0, None)
    for hx, sx in ((x0, 1), (x1, -1)):                    # charnière côté gauche ou droit
        for sy in (1, -1):                                # battant côté bas ou haut
            for hy in (y1 if sy > 0 else y0, yc):
                s = _arc_score(ink_d, hx, hy, g, sx, sy)
                if s > best[0]:
                    best = (s, dict(hinge=(hx, hy), sx=sx, sy=sy))
    if best[0] >= 0.6:
        return "porte", min(0.95, 0.5 + 0.5 * best[0]), best[1]
    # battant sans arc : un trait perpendiculaire au mur, long comme l'ouverture, qui part d'un montant
    lead = (0.0, None)
    for hx, sx in ((x0, 1), (x1, -1)):
        for sy in (1, -1):
            hy = y1 if sy > 0 else y0
            n_ok = tot = 0
            for t in np.linspace(0.15, 0.95, 24):
                yy = int(round(hy + sy * (t * g + 2)))
                tot += 1
                if 0 <= yy < ink_d.shape[0] and 0 <= hx + sx * 2 < ink_d.shape[1] and ink_d[yy, int(hx + sx * 2)]:
                    n_ok += 1
            sc = n_ok / max(1, tot)
            if sc > lead[0]:
                lead = (sc, dict(hinge=(hx, hy), sx=sx, sy=sy))
    if lead[0] >= 0.9:
        return "porte", 0.65, lead[1]
    inner = ink[y0 + 1:y1 - 1, x0 + int(0.08 * g):x1 - int(0.08 * g)]
    if inner.size:
        rows = (inner > 0).mean(axis=1)
        lines = int((rows >= 0.75).sum())
        if lines >= 1 or (inner > 0).mean() >= 0.08:
            return "fenetre", 0.75 if lines else 0.55, {}
    return "ouverture", 0.5, {}


def _interior_windows(raw: np.ndarray, wall: np.ndarray, wt: float, real_px: float):
    """Double trait : fenêtre = traits fins intérieurs à l'épaisseur du mur, sur 0,4 à 3 m (le mur peut rester continu)."""
    out = []
    for (x0, x1, y0, y1) in _segments(wall, wt):
        h = y1 - y0
        ya, yb = y0 + max(2, int(0.3 * h)), y1 - max(2, int(0.3 * h))
        if yb <= ya:
            continue
        band = raw[ya:yb, x0:x1] > 0
        runs = []
        for r in range(band.shape[0]):
            row = np.concatenate([[0], band[r].astype(np.int8), [0]])
            d = np.diff(row)
            for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                if b - a >= 0.4 * real_px:
                    runs.append((a, b))
        runs.sort()
        merged = []
        for a, b in runs:
            if merged and a <= merged[-1][1] + 0.1 * real_px:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] + 1)
            else:
                merged.append((a, b, 1))
        for a, b, nrows in merged:
            L = b - a
            if L > 3.0 * real_px or L >= 0.95 * (x1 - x0):          # mur « à isolant » sur toute sa longueur : pas une fenêtre
                continue
            out.append((x0 + a, y0, x0 + b, y1))
    return out


def _iou_rect(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect_openings(ink: np.ndarray, mm_px: float, scale: float, wt: float | None = None, raw: np.ndarray | None = None):
    """Renvoie (liste d'ouvertures, épaisseur de mur px). Chaque ouverture :
    dict(kind, conf, rect=(x0,y0,x1,y1) px, orient='h'|'v', width_mm (réel), hinge, sx, sy)."""
    if wt is None:
        wt = wall_thickness(ink, mm_px)
    if wt is None:
        return [], None
    k = max(3, int(0.6 * wt)) | 1
    wall = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    real_px = 1000.0 / (mm_px * scale)                    # 1 m réel en pixels
    gmin, gmax = 0.5 * real_px, 3.0 * real_px
    ink_d = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    res = []
    for orient in ("h", "v"):
        if orient == "h":
            w_, i_, d_ = wall, ink, ink_d
            r_ = raw
        else:
            w_, i_, d_ = (np.ascontiguousarray(wall.T), np.ascontiguousarray(ink.T), np.ascontiguousarray(ink_d.T))
            r_ = np.ascontiguousarray(raw.T) if raw is not None else None
        win = _interior_windows(r_, w_, wt, real_px) if r_ is not None else []
        rects = [(r, True) for r in win] + [(r, False) for r in _gaps(w_, wt, gmin, gmax)]
        kept: list = []
        for rect, interior in rects:
            if not interior and any(_iou_rect(rect, q) > 0.25 for q in win):
                continue                                   # déjà trouvée comme fenêtre (traits intérieurs)
            if interior:
                kind, conf, extra = "fenetre", 0.7, {}
            else:
                kind, conf, extra = _classify(i_, d_, rect, wt)
            x0, y0, x1, y1 = rect
            o = dict(kind=kind, conf=round(conf, 2), orient=orient, width_mm=round((x1 - x0) * mm_px * scale),
                     cut=bool(interior))
            if orient == "h":
                o["rect"] = (x0, y0, x1, y1)
                if extra:
                    o.update(hinge=extra["hinge"], sx=extra["sx"], sy=extra["sy"])
            else:                                          # retour dans le repère image
                o["rect"] = (y0, x0, y1, x1)
                if extra:
                    hx, hy = extra["hinge"]
                    o.update(hinge=(hy, hx), sx=extra["sy"], sy=extra["sx"])
            res.append(o)
    return res, wt


def zone_of(o: dict, wt: float):
    """Boîtes (x0,y0,x1,y1) où les tracés fins appartiennent à cette ouverture."""
    x0, y0, x1, y1 = o["rect"]
    m = 0.8 * wt
    boxes = [(x0 - m, y0 - m, x1 + m, y1 + m)]
    if o["kind"] == "porte" and "hinge" in o:
        hx, hy = o["hinge"]
        r = 1.18 * max(x1 - x0, y1 - y0)
        sx, sy = o["sx"], o["sy"]
        bx = (hx, hx + sx * r) if sx > 0 else (hx - r, hx)
        by = (hy, hy + sy * r) if sy > 0 else (hy - r, hy)
        boxes.append((min(bx) - m, min(by) - m, max(bx) + m, max(by) + m))
    return boxes


def _inside(pts: np.ndarray, boxes) -> bool:
    for (a, b, c, d) in boxes:
        if (pts[:, 0] >= a).all() and (pts[:, 0] <= c).all() and (pts[:, 1] >= b).all() and (pts[:, 1] <= d).all():
            return True
    return False


def assign_openings(polys_px, openings, wt: float):
    """polys_px : [(pts, closed, width_px)] -> {index: calque} pour les tracés fins contenus dans une zone d'ouverture."""
    out = {}
    zones = [(o, zone_of(o, wt)) for o in openings]
    for i, (pts, closed, wpx) in enumerate(polys_px):
        if wpx >= 0.75 * wt:
            continue
        for o, boxes in zones:
            if _inside(np.asarray(pts, float), boxes):
                out[i] = WINDOW_LAYER if o["kind"] == "fenetre" else DOOR_LAYER
                break
    return out


# ---------------------------------------------------------------- escaliers
def detect_stairs(polys_px, mm_px: float, scale: float, wt: float | None, min_steps: int = 4):
    """Renvoie une liste d'escaliers : dict(idx=[indices], rect=[4 points px], marches=n, conf)."""
    cand = []
    for i, (pts, closed, wpx) in enumerate(polys_px):
        p = np.asarray(pts, float)
        if closed or len(p) < 2 or (wt and wpx >= 0.75 * wt):
            continue
        v = p[-1] - p[0]
        L = float(np.hypot(*v))
        path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if L < 1 or path > 1.08 * L:                       # pas droit
            continue
        Lr = L * mm_px * scale / 1000.0
        if not 0.6 <= Lr <= 3.2:
            continue
        ang = (np.degrees(np.arctan2(v[1], v[0]))) % 180.0
        cand.append((i, p[0], p[-1], L, ang))
    stairs, used = [], set()
    cand.sort(key=lambda c: c[4])
    for a in range(len(cand)):
        grp = [c for c in cand if min(abs(c[4] - cand[a][4]), 180 - abs(c[4] - cand[a][4])) <= 4.0
               and abs(c[3] - cand[a][3]) <= 0.15 * cand[a][3] and c[0] not in used]
        if len(grp) < min_steps:
            continue
        th = np.radians(cand[a][4])
        t = np.array([np.cos(th), np.sin(th)])
        n = np.array([-np.sin(th), np.cos(th)])
        cen = np.array([((c[1] + c[2]) / 2) for c in grp])
        off = cen @ n
        tan = cen @ t
        order = np.argsort(off)
        grp = [grp[j] for j in order]
        off, tan = off[order], tan[order]
        run = [0]
        runs = []
        for j in range(1, len(grp)):
            sp = (off[j] - off[j - 1]) * mm_px * scale / 1000.0
            ok = 0.17 <= sp <= 0.42 and abs(tan[j] - tan[j - 1]) <= 0.3 * cand[a][3]
            if ok and len(run) >= 2:
                prev = (off[run[-1]] - off[run[-2]])
                ok = abs((off[j] - off[j - 1]) - prev) <= 0.3 * prev
            if ok:
                run.append(j)
            else:
                if len(run) >= min_steps:
                    runs.append(run)
                run = [j]
        if len(run) >= min_steps:
            runs.append(run)
        for r in runs:
            members = [grp[j] for j in r]
            pts = np.array([q for c in members for q in (c[1], c[2])])
            ext_t = (pts @ t).min(), (pts @ t).max()
            ext_n = (pts @ n).min(), (pts @ n).max()
            rect = [t * ext_t[a_] + n * ext_n[b_] for a_, b_ in ((0, 0), (1, 0), (1, 1), (0, 1))]
            idx = [c[0] for c in members]
            used.update(idx)
            conf = min(0.9, 0.5 + 0.08 * len(idx))
            stairs.append(dict(idx=idx, rect=[tuple(map(float, q)) for q in rect], marches=len(idx), conf=round(conf, 2)))
    return stairs


# ---------------------------------------------------------------- escaliers en éventail (marches rayonnantes)
def detect_fan_stairs(polys_px, mm_px: float, scale: float, wt: float | None, exclude=(), min_steps: int = 5):
    """Quart tournant / hélicoïdal : ≥ 5 traits droits dont les prolongements passent par un même centre, régulièrement
    répartis en angle. Renvoie des dict(idx, rect=[polygone convexe], marches, conf, centre)."""
    cand = []                                  # (indice du tracé, début, fin, longueur) : un SEGMENT droit par entrée
    for i, (pts, closed, wpx) in enumerate(polys_px):
        if i in exclude or (wt and wpx >= 0.75 * wt):
            continue
        p = np.asarray(pts, float)
        if len(p) < 2:
            continue
        if closed:
            p = np.vstack([p, p[:1]])
        for a_, b_ in zip(p[:-1], p[1:]):
            L = float(np.hypot(*(b_ - a_)))
            Lr = L * mm_px * scale / 1000.0
            if 0.4 <= Lr <= 2.5:
                cand.append((i, a_, b_, L))
    if len(cand) < min_steps or len(cand) > 900:
        return []
    P0 = np.array([c[1] for c in cand]); P1 = np.array([c[2] for c in cand])
    D = P1 - P0
    Ls = np.array([c[3] for c in cand])
    U = D / Ls[:, None]
    cell = 0.3 * 1000.0 / (mm_px * scale)                 # 30 cm
    votes: dict = {}
    n = len(cand)
    for a in range(n):
        for b in range(a + 1, n):
            cr = U[a, 0] * U[b, 1] - U[a, 1] * U[b, 0]
            if abs(cr) < np.sin(np.radians(6)) or abs(cr) > np.sin(np.radians(70)):
                continue
            w = P0[b] - P0[a]
            t = (w[0] * U[b, 1] - w[1] * U[b, 0]) / cr
            pt = P0[a] + t * U[a]
            key = (int(round(pt[0] / cell)), int(round(pt[1] / cell)))
            votes.setdefault(key, set()).update((a, b))
    stairs, used = [], set(exclude)
    for key, members in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_steps:
            break
        c = np.array([key[0] * cell, key[1] * cell])
        mem = []
        for i in members:
            if cand[i][0] in used:
                continue
            # le prolongement passe près du centre, qui est HORS du segment (marche qui rayonne)
            dpt = c - P0[i]
            u = float(dpt @ U[i])
            dist = abs(dpt[0] * U[i, 1] - dpt[1] * U[i, 0])
            if dist <= max(1.5 * cell, 0.08 * Ls[i]) and (u < -0.05 * Ls[i] or u > 1.05 * Ls[i]):
                mem.append(i)
        if len(mem) < min_steps:
            continue
        ang = np.sort([np.degrees(np.arctan2(*(((P0[i] + P1[i]) / 2 - c)[::-1]))) % 360 for i in mem])
        gaps = np.diff(ang)
        gaps = gaps[gaps > 0.5]
        if len(gaps) < min_steps - 1 or np.median(gaps) < 4 or np.median(gaps) > 40 or gaps.std() > 0.6 * gaps.mean():
            continue
        pts = np.vstack([np.vstack([P0[i], P1[i]]) for i in mem]).astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        idx = sorted({cand[i][0] for i in mem})
        used.update(idx)
        stairs.append(dict(idx=idx, rect=[tuple(map(float, q)) for q in hull], marches=len(idx),
                           conf=round(min(0.85, 0.45 + 0.07 * len(idx)), 2), kind="eventail"))
    return stairs


# ---------------------------------------------------------------- découpe des murs au droit des fenêtres
def _clip_segment(a, b, rect):
    """Partie de [a,b] à l'intérieur du rectangle : (t0, t1) ou None (Liang-Barsky)."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
    return (t0, t1) if t0 < t1 else None


def cut_polyline(pts: np.ndarray, closed: bool, rects) -> list[np.ndarray]:
    """Retire les portions de polyligne situées dans les rectangles. Renvoie la liste des morceaux (≥ 2 points)."""
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    pieces: list[list] = [[p[0]]]
    for a, b in zip(p[:-1], p[1:]):
        cuts = []
        for r in rects:
            c = _clip_segment(a, b, r)
            if c:
                cuts.append(c)
        if not cuts:
            pieces[-1].append(b)
            continue
        cuts.sort()
        merged = [list(cuts[0])]
        for t0, t1 in cuts[1:]:
            if t0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1])
        pos = 0.0
        for t0, t1 in merged:
            if t0 > pos:
                pieces[-1].append(a + (b - a) * t0)
            pieces.append([a + (b - a) * t1]) if t1 < 1.0 else pieces.append([])
            pos = t1
        if pos < 1.0:
            pieces[-1].append(b)
        elif not pieces[-1]:
            pass
        # début du morceau suivant déjà posé ; si l'extrémité b est coupée, le morceau suivant commencera au segment suivant
        if pieces and not pieces[-1]:
            pieces[-1] = [b]
    out = [np.array(q) for q in pieces if len(q) >= 2]
    return out


# ---------------------------------------------------------------- façade : une baie sans symbole sur l'enveloppe = fenêtre
def tag_envelope(ink: np.ndarray, openings: list, wt: float, mm_px: float, scale: float):
    """Pour les ouvertures « nues » (ni arc, ni battant, ni trait de fenêtre) : si un côté donne sur l'extérieur du bâtiment
    et l'autre sur l'intérieur, c'est une fenêtre (ou une porte d'entrée) ; sinon baie intérieure / porte coulissante / double porte."""
    if not openings:
        return
    H, W = ink.shape
    barrier = ink.copy()
    for o in openings:
        x0, y0, x1, y1 = o["rect"]
        cv2.rectangle(barrier, (x0, y0), (x1, y1), 255, -1)
    free = (barrier == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) - {0}
    ext = np.isin(lab, list(border)) if border else np.zeros_like(free, bool)
    off = int(round(1.4 * wt))
    for o in openings:
        if o["kind"] != "ouverture":
            continue
        x0, y0, x1, y1 = o["rect"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if o["orient"] == "h":
            pa, pb = (cx, y0 - off), (cx, y1 + off)
        else:
            pa, pb = (x0 - off, cy), (x1 + off, cy)
        def is_ext(p):
            x, y = p
            return bool(ext[y, x]) if 0 <= x < W and 0 <= y < H else True
        ea, eb = is_ext(pa), is_ext(pb)
        w_m = o["width_mm"] / 1000.0
        if ea != eb:
            o["envelope"] = True
            if w_m <= 2.6:
                o["kind"] = "fenetre"
                o["conf"] = 0.5
                o["why"] = "baie sur la façade sans symbole : fenêtre (ou porte d'entrée sans arc) – à vérifier"
        else:
            o["envelope"] = False
            o["why"] = ("baie intérieure sans symbole : porte coulissante, porte double sans battant ou passage"
                        if w_m >= 0.7 else "passage étroit sans symbole")


# ---------------------------------------------------------------- escaliers dessinés par arcs concentriques
def _fit_circle(p: np.ndarray):
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(p))]
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0)))
    res = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
    return cx, cy, r, res


def detect_arc_stairs(polys_px, mm_px: float, scale: float, wt: float | None, exclude=(), min_arcs: int = 3):
    """Marches courbes : ≥ 3 arcs concentriques, rayons régulièrement espacés de 17 à 42 cm."""
    real_px = 1000.0 / (mm_px * scale)
    arcs = []
    for i, (pts, closed, wpx) in enumerate(polys_px):
        if i in exclude or closed or (wt and wpx >= 0.75 * wt):
            continue
        p = np.asarray(pts, float)
        if len(p) < 4:
            continue
        chord = float(np.hypot(*(p[-1] - p[0])))
        path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        if path < 0.5 * real_px or path < 1.08 * chord:                 # trop court, ou droit
            continue
        f = _fit_circle(p)
        if f is None:
            continue
        cx, cy, r, res = f
        if not (0.4 * real_px <= r <= 5.0 * real_px) or res > 0.06 * r:
            continue
        arcs.append((i, np.array([cx, cy]), r))
    out, used = [], set()
    for a in arcs:
        if a[0] in used:
            continue
        grp = [b for b in arcs if b[0] not in used and np.hypot(*(a[1] - b[1])) <= 0.3 * real_px]
        if len(grp) < min_arcs:
            continue
        grp.sort(key=lambda t: t[2])
        radii = np.array([g[2] for g in grp])
        sp = np.diff(radii) / real_px
        ok = [(0.17 <= s <= 0.42) for s in sp]
        # plus longue suite régulière
        best, cur = [], [0]
        for j, good in enumerate(ok, 1):
            if good and (len(cur) < 2 or abs(sp[j - 1] - sp[j - 2]) <= 0.35 * sp[j - 2]):
                cur.append(j)
            else:
                best = cur if len(cur) > len(best) else best
                cur = [j]
        best = cur if len(cur) > len(best) else best
        if len(best) >= min_arcs:
            members = [grp[j] for j in best]
            idx = [m[0] for m in members]
            used.update(idx)
            pts = np.vstack([np.asarray(polys_px[i][0], float) for i in idx]).astype(np.float32)
            hull = cv2.convexHull(pts).reshape(-1, 2)
            out.append(dict(idx=idx, rect=[tuple(map(float, q)) for q in hull], marches=len(idx) - 1,
                            conf=round(min(0.8, 0.4 + 0.1 * len(idx)), 2), kind="arcs"))
    return out
