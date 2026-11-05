"""Cotes : relie chaque nombre écrit sur le plan à la ligne de cote qui l'accompagne.

Intérêt : la valeur écrite (ex. « 8.25 ») et la longueur dessinée donnent l'ÉCHELLE réelle du plan. On la compare à
l'échelle demandée (--scale) : une échelle fausse fausse tout (surfaces, 3D). Avec `--scale auto`, elle est utilisée.
Les lignes de cote sont reclassées sur ARC-COTES, leur texte aussi.
"""
from __future__ import annotations

import re

import numpy as np

COTE_LAYER = "ARC-COTES"
STD_METRIC = [1, 2, 5, 10, 20, 25, 50, 75, 100, 125, 150, 200, 250, 500, 1000]
STD_IMPERIAL = [12, 16, 24, 32, 48, 64, 96, 128, 192, 240, 384]       # 1"=1', 3/4", 1/2", 3/8", 1/4", 3/16", 1/8", 3/32", 1/16"...
STD_SCALES = sorted(set(STD_METRIC) | set(STD_IMPERIAL))
_FT = re.compile(r"^(\d{1,3})\s*['’`´‘]\s*[-–]?\s*(\d{1,2}(?:[.]\d)?)?\s*[\"”'’‘]{0,2}$")
_FT2 = re.compile(r"^(\d{1,3})\s*[-–]\s*(\d{1,2})$")                                  # « 21-0 » (pieds-pouces sans signes)
_ROOM = re.compile(r"(\d{1,3})\s*[-–'.]\s*(\d{1,2})\s*[x×*=nX]\s*(\d{1,3})\s*[-–'.]\s*(\d{1,2})")
_ROOM_M = re.compile(r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m)?\s*[x×*X]\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m)?$")
_AREA = re.compile(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:m\s*[²2]|m\s*2|sq\.?\s*m)", re.I)
_NUM = re.compile(r"^(\d{1,6}(?:[.,]\d{1,3})?)\s*(mm|cm|m)?$", re.I)


def parse_value_sys(txt: str):
    """-> (valeurs possibles en mm réels, système 'imp' | 'met'). Pieds-pouces : 26'-0\", 26'0, 21-0."""
    t = txt.strip()
    m = _FT.match(t)
    if m:
        ft, inch = int(m.group(1)), float(m.group(2) or 0)
        return [ft * 304.8 + inch * 25.4], "imp"
    m = _FT2.match(t)
    if m and int(m.group(2)) < 12:
        return [int(m.group(1)) * 304.8 + int(m.group(2)) * 25.4], "imp"
    return parse_value(txt), "met"


def parse_room_label(txt: str):
    """Libellé de pièce : « 11-4 x 10-0 » (pieds-pouces), « 3,50 x 4,20 » (m) ou « 12,5 m² ».
    -> dict(kind='dims', a_mm, b_mm, sys) | dict(kind='area', m2) | None."""
    t = txt.strip()
    m = _ROOM.search(t)
    if m:
        a = int(m.group(1)) * 304.8 + int(m.group(2)) * 25.4
        b = int(m.group(3)) * 304.8 + int(m.group(4)) * 25.4
        if int(m.group(2)) < 12 and int(m.group(4)) < 12 and a > 600 and b > 600:
            return dict(kind="dims", a_mm=max(a, b), b_mm=min(a, b), sys="imp")
    m = _ROOM_M.search(t.replace(" ", ""))
    if m:
        a = float(m.group(1).replace(",", ".")) * 1000.0
        b = float(m.group(2).replace(",", ".")) * 1000.0
        if 1000 <= a <= 30000 and 1000 <= b <= 30000:
            return dict(kind="dims", a_mm=max(a, b), b_mm=min(a, b), sys="met")
    m = _AREA.search(t)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 1.0 <= v <= 500.0:
            return dict(kind="area", m2=v, sys="met")
    return None


def parse_value(txt: str):
    """-> liste de valeurs possibles en mm réels, la plus plausible d'abord, ou []."""
    t = txt.strip().replace(" ", "").replace("O", "0").replace("o", "0") if re.search(r"\d", txt) else txt
    t = t.replace("'", ".")
    m = _NUM.match(t)
    if not m:
        return []
    raw = m.group(1)
    v = float(raw.replace(",", "."))
    unit = (m.group(2) or "").lower()
    if unit:
        return [v * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[unit]]
    if re.search(r"[.,]", raw):                      # « 3.50 » : mètres, plus rarement cm
        return [v * 1000.0, v * 10.0]
    if v >= 1000:                                    # « 3500 » : mm
        return [v, v * 10.0]
    return [v * 10.0, v, v * 1000.0]                 # « 350 » : cm d'abord


def find_dimensions(phrases: list, polys_px: list, mm_px: float, wt: float | None = None):
    """Associe les phrases numériques à des lignes droites parallèles proches.
    polys_px : [(pts, closed, width_px)]. Renvoie une liste de dict(phrase, poly, L_px, cands=[mm...])."""
    lines = []
    for i, (pts, closed, wpx) in enumerate(polys_px):
        p = np.asarray(pts, float)
        if closed or len(p) < 2 or (wt and wpx >= 0.75 * wt):
            continue
        v = p[-1] - p[0]
        L = float(np.hypot(*v))
        if L < 8 or float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) > 1.04 * L:
            continue
        lines.append((i, p[0], p[-1], L, np.degrees(np.arctan2(v[1], v[0])) % 180.0))
    out = []
    used: set[int] = set()
    for pi, ph in enumerate(phrases):
        vals, sys_ = parse_value_sys(ph["text"])
        if not vals:
            continue
        th = ph.get("th", ph["h"])
        cx, cy = ph["x"] + ph["w"] / 2, ph["y"] + ph["h"] / 2
        tang = 0.0 if ph["angle"] == 0 else 90.0
        best = None
        for i, a, b, L, ang in lines:
            if i in used or L < 3.0 * th:
                continue
            da = min(abs(ang - tang), 180 - abs(ang - tang))
            if da > 6.0:
                continue
            d = b - a
            t = np.array([d[0], d[1]]) / L
            n = np.array([-t[1], t[0]])
            rel = np.array([cx, cy]) - a
            u, v = float(rel @ t), float(rel @ n)
            if not (-0.15 * L <= u <= 1.15 * L) or abs(v) > 2.6 * th + 4:
                continue
            if best is None or abs(v) < best[0]:
                best = (abs(v), i, L)
        if best is None:
            best2 = _split_line(ph, th, lines, used)
            if best2 is None:
                continue
            for j in best2[1]:
                used.add(j)
            out.append(dict(phrase=pi, poly=best2[1][0], polys=best2[1], L_px=best2[2], cands=vals, text=ph["text"],
                            sys=sys_))
            continue
        used.add(best[1])
        out.append(dict(phrase=pi, poly=best[1], polys=[best[1]], L_px=best[2], cands=vals, text=ph["text"], sys=sys_))
    return out


def _split_line(ph, th, lines, used):
    """Cote dont le texte INTERROMPT la ligne : deux tronçons colinéaires de part et d'autre du texte."""
    cx, cy = ph["x"] + ph["w"] / 2, ph["y"] + ph["h"] / 2
    horiz = ph["angle"] == 0
    tang = 0.0 if horiz else 90.0
    left, right = None, None
    for i, a, b, L, ang in lines:
        if i in used or min(abs(ang - tang), 180 - abs(ang - tang)) > 6.0:
            continue
        p0, p1 = (a, b) if (a[0] <= b[0] if horiz else a[1] <= b[1]) else (b, a)
        perp = abs(p0[1] - cy) if horiz else abs(p0[0] - cx)
        perp2 = abs(p1[1] - cy) if horiz else abs(p1[0] - cx)
        if max(perp, perp2) > 1.2 * th:
            continue
        lo_t = (p0[0] if horiz else p0[1]); hi_t = (p1[0] if horiz else p1[1])
        c = cx if horiz else cy
        half = (ph["w"] if horiz else ph["h"]) / 2
        if hi_t <= c - half * 0.6 and c - half - hi_t <= 2.5 * th:
            if left is None or hi_t > left[2]:
                left = (i, lo_t, hi_t)
        elif lo_t >= c + half * 0.6 and lo_t - (c + half) <= 2.5 * th:
            if right is None or lo_t < right[1]:
                right = (i, lo_t, hi_t)
    if left and right:
        return (0.0, [left[0], right[0]], right[2] - left[1])
    return None


def estimate_scale(dims: list, mm_px: float):
    """Échelle (dénominateur) la plus cohérente avec les cotes. Renvoie dict ou None.
    Chaque cote propose, pour chaque unité possible, une échelle ; on retient celle qui tombe le plus près d'une échelle
    normalisée, puis la médiane si au moins 3 cotes concordent (±6 %)."""
    props = []
    systems = []
    for d in dims:
        paper = d["L_px"] * mm_px
        best = None
        for mm in d["cands"]:
            s = mm / paper
            if s < 1.5 or s > 2000:
                continue
            stds = STD_IMPERIAL if d.get("sys") == "imp" else STD_METRIC
            std = min(stds, key=lambda x: abs(np.log(s / x)))
            err = abs(np.log(s / std))
            if best is None or err < best[0]:
                best = (err, s, std)
        if best and best[0] < 0.12:
            props.append(best[1])
            systems.append(d.get("sys", "met"))
    if len(props) < 3:
        return None
    med = float(np.median(props))
    agree = [p for p in props if abs(p / med - 1) <= 0.06]
    if len(agree) < max(3, int(0.6 * len(props))):
        return None
    s = float(np.median(agree))
    imp = sum(1 for p_, sy in zip(props, systems) if sy == "imp" and abs(p_ / med - 1) <= 0.06) > len(agree) / 2
    stds = STD_IMPERIAL if imp else STD_METRIC
    std = min(stds, key=lambda x: abs(np.log(s / x)))
    snapped = std if abs(s / std - 1) <= 0.06 else s
    return dict(scale=float(snapped), measured=round(s, 1), n=len(agree), of=len(props),
                systeme="pieds-pouces" if imp else "métrique")


def room_scale(phrases: list, ink: np.ndarray, mm_px: float, wt: float | None):
    """Échelle d'après les DIMENSIONS DE PIÈCES écrites (« 11-4 x 10-0 », « 3,50 x 4,20 », « 12,5 m² ») comparées à la
    pièce dessinée qui entoure le libellé. Les passages de porte sont bouchés par fermeture (taille liée à l'épaisseur des
    murs, donc sans dépendre de l'échelle). Renvoie dict comme estimate_scale, ou None."""
    import cv2
    labs = [(i, parse_room_label(p["text"])) for i, p in enumerate(phrases)]
    labs = [(i, r) for i, r in labs if r]
    if not labs:
        return None
    k = int(max(5, 5 * (wt or 20)))
    barrier = ink.copy()
    barrier = cv2.bitwise_or(barrier, cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))))
    barrier = cv2.bitwise_or(barrier, cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))))
    free = (barrier == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = ink.shape
    props, systems, used = [], [], set()
    for i, r in labs:
        p = phrases[i]
        cx, cy = int(p["x"] + p["w"] / 2), int(p["y"] + p["h"] / 2)
        c = 0
        rad = int(2 * p.get("th", p["h"]))
        sub = lab[max(0, cy - rad):cy + rad + 1, max(0, cx - rad):cx + rad + 1]
        ids = sub[sub > 0]
        if len(ids):
            c = int(np.bincount(ids).argmax())
        if c == 0 or c in used:
            continue
        x, y, w, h, a = st[c]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue                                   # pièce non fermée (déborde sur l'extérieur)
        comp = (lab == c).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cx0, cy0), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        L, S = max(rw, rh), min(rw, rh)
        if S < 10 or a < 0.8 * L * S:
            continue                                   # pièce non rectangulaire : mesure peu fiable
        used.add(c)
        if r["kind"] == "dims":
            s1 = r["a_mm"] / (L * mm_px)
            s2 = r["b_mm"] / (S * mm_px)
            if abs(s1 / s2 - 1) > 0.15:
                continue
            props.append(float(np.sqrt(s1 * s2)))
        else:
            props.append(float(np.sqrt(r["m2"] * 1e6 / (a * mm_px * mm_px))))
        systems.append(r["sys"])
    return _agree(props, systems)


def _agree(props: list, systems: list):
    if len(props) < 3:
        return None
    med = float(np.median(props))
    agree = [p for p in props if abs(p / med - 1) <= 0.08]
    if len(agree) < max(3, int(0.6 * len(props))):
        return None
    s = float(np.median(agree))
    imp = sum(1 for p_, sy in zip(props, systems) if sy == "imp" and abs(p_ / med - 1) <= 0.08) > len(agree) / 2
    stds = STD_IMPERIAL if imp else STD_METRIC
    std = min(stds, key=lambda x: abs(np.log(s / x)))
    snapped = std if abs(s / std - 1) <= 0.08 else s
    return dict(scale=float(snapped), measured=round(s, 1), n=len(agree), of=len(props),
                systeme="pieds-pouces" if imp else "métrique", source="dimensions de pièces")
