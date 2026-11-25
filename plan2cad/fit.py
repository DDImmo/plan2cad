"""Ajustement de primitives CAO : cercles et arcs vrais à la place des polylignes à facettes.

Idée reprise de raster2cad (primitives CAO : lignes, cercles) et de vectorizer.ai (on ajuste des formes entières,
cercles / arcs, plutôt que de décrire un contour point par point) : une suite de sommets qui suit un cercle devient
UN objet ARC (ou CERCLE si la polyligne se ferme), les segments droits qui l'entourent restent des polylignes.
Les arcs de porte, murs courbes et poteaux ronds restent ainsi éditables dans le logiciel de CAO.
"""
from __future__ import annotations

import numpy as np

MIN_VERTS = 4            # un arc ajusté sur moins de 4 sommets n'est pas fiable
MIN_SWEEP = 20.0         # degrés
MAX_REL_RMS = 0.02       # écart quadratique moyen / rayon
MAX_TURN_DEG = 40.0      # virage maximal à un sommet d'arc


def fit_circle(P: np.ndarray):
    """Cercle des moindres carrés (Kåsa) : (cx, cy, r, rms) ou None."""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(P))]
    try:
        sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(sol[0]), float(sol[1])
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return cx, cy, r, rms


def _turn_signs(P):
    d = np.diff(P, axis=0)
    cr = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    return np.sign(cr)


def _good(P, tol_abs):
    """Cercle plausible pour ce groupe de sommets ? Renvoie le cercle ou None."""
    if len(P) < MIN_VERTS:
        return None
    s = _turn_signs(P)
    s = s[s != 0]
    if len(s) < len(P) - 2 or not (np.all(s > 0) or np.all(s < 0)):      # un arc tourne toujours du même côté
        return None
    # virage par sommet : un arc échantillonné tourne peu à chaque sommet ; 4 coins d'un rectangle sont cocycliques
    d = np.diff(P, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    if len(turn) and turn.max() > np.radians(MAX_TURN_DEG):
        return None
    c = fit_circle(P)
    if c is None:
        return None
    cx, cy, r, rms = c
    if rms > max(MAX_REL_RMS * r, tol_abs):
        return None
    # le cercle ajusté doit rester près de CHAQUE corde (pas seulement des sommets) : flèche réelle = flèche de l'arc
    mid = (P[:-1] + P[1:]) / 2
    if np.abs(np.hypot(mid[:, 0] - cx, mid[:, 1] - cy) - r).max() > 0.12 * r:
        return None
    if np.hypot(*(P[-1] - P[0])) > 2 * r * 1.0001:
        return None
    # pas de côtés démesurés : sur un arc, les cordes successives sont du même ordre
    L = np.hypot(*np.diff(P, axis=0).T)
    if L.max() > 3.5 * np.median(L) + 1e-9 and len(L) > 3:
        return None
    return c


def _angles(P, c):
    cx, cy, r, _ = c
    a = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    return a


def fit_polyline(pts: np.ndarray, closed: bool, tol_abs: float):
    """Découpe une polyligne en morceaux : [("poly", pts) | ("arc", cx, cy, r, a0_deg, a1_deg) | ("circle", cx, cy, r)].
    tol_abs : écart toléré (unités du dessin) sous lequel un ajustement est accepté même si rms > 2 % du rayon."""
    P = np.asarray(pts, float)
    n = len(P)
    if closed and n >= 6:
        c = _good(np.vstack([P, P[:1]]) if False else P, tol_abs)
        if c is not None:
            a = _angles(np.vstack([P, P[:1]]), c)
            if abs(a[-1] - a[0]) > np.radians(330) and abs(abs(a[-1] - a[0]) - 2 * np.pi) < np.radians(40):
                return [("circle", c[0], c[1], c[2])]
    if n < MIN_VERTS:
        return [("poly", P)]
    pieces = []
    i = 0
    last_end = 0
    while i <= n - MIN_VERTS:
        j = i + MIN_VERTS
        c = _good(P[i:j], tol_abs)
        if c is None:
            i += 1
            continue
        best = (j, c)
        while j < n:
            c2 = _good(P[i:j + 1], tol_abs)
            if c2 is None:
                break
            j += 1
            best = (j, c2)
        j, c = best
        a = _angles(P[i:j], c)
        sweep = abs(np.degrees(a[-1] - a[0]))
        if sweep < MIN_SWEEP or sweep > 350:
            i += 1
            continue
        if i > last_end:
            pieces.append(("poly", P[last_end:i + 1]))
        cx, cy, r, _ = c
        a0, a1 = np.degrees(a[0]), np.degrees(a[-1])
        if a1 < a0:                                    # DXF : toujours dans le sens anti-horaire
            a0, a1 = a1, a0
        pieces.append(("arc", cx, cy, r, a0 % 360.0, a1 % 360.0 if a1 % 360.0 != a0 % 360.0 else a1))
        last_end = j - 1
        i = j - 1
    if not pieces:
        return [("poly", P)]
    if last_end < n - 1:
        pieces.append(("poly", P[last_end:]))
    return pieces


def to_entities(polys, texts=None, tol_abs: float = 0.0, layers_skip=("ARC-TEXTE",)):
    """Polylignes -> entités DXF avec arcs / cercles vrais. Renvoie (entités, nb_arcs, nb_cercles)."""
    ents, n_arc, n_circ = [], 0, 0
    for p in polys:
        pts = np.asarray(p.pts, float)
        if len(pts) < 3 or p.layer in layers_skip:
            ents.append(("polyline", p.layer, p.pts, p.closed))
            continue
        pieces = fit_polyline(pts, p.closed, tol_abs)
        if len(pieces) == 1 and pieces[0][0] == "poly":
            ents.append(("polyline", p.layer, p.pts, p.closed))
            continue
        for pc in pieces:
            if pc[0] == "poly":
                if len(pc[1]) >= 2:
                    ents.append(("polyline", p.layer, pc[1], False))
            elif pc[0] == "arc":
                ents.append(("arc", p.layer, (pc[1], pc[2]), pc[3], pc[4], pc[5]))
                n_arc += 1
            else:
                ents.append(("circle", p.layer, (pc[1], pc[2]), pc[3]))
                n_circ += 1
    return ents, n_arc, n_circ


def refit_chain(pts: np.ndarray, closed: bool, step_deg: float = 4.0, min_px: float = 6.0):
    """Ajuste un cercle sur les pixels BRUTS d'une branche de squelette (des centaines de points : précision sous-pixel,
    bien meilleure que les quelques sommets de Douglas-Peucker). Si la branche est un cercle ou un arc net, renvoie
    la branche ré-échantillonnée exactement sur le cercle ajusté (extrémités conservées) ; sinon None."""
    P = np.asarray(pts, float)
    if len(P) < 12:
        return None
    c = fit_circle(P)
    if c is None:
        return None
    cx, cy, r, rms = c
    if r < min_px or r > 20000:
        return None
    if rms > min(3.0, max(1.0, 0.004 * r)):               # bruit de squelette ≈ 0,3-0.7 px ; au-delà ce n'est pas un cercle
        return None
    a = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    sweep = float(a[-1] - a[0])
    if np.any(np.diff(np.sign(np.diff(a))) != 0) and np.ptp(a) > abs(sweep) * 1.05 + 0.2:
        return None                                       # le tracé fait demi-tour : pas un arc
    if closed:
        if abs(abs(sweep) - 2 * np.pi) > np.radians(25) and np.ptp(a) < np.radians(330):
            return None
        n = max(12, int(360.0 / step_deg))
        t = a[0] + np.linspace(0, 2 * np.pi * (1 if sweep >= 0 else -1), n, endpoint=False)
        return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
    if abs(np.degrees(sweep)) < MIN_SWEEP:
        return None
    if r * abs(sweep) < 40.0 or r * (1 - np.cos(abs(sweep) / 2)) < 2.5:    # trop court / trop plat : c'est un segment
        return None
    n = max(4, int(abs(np.degrees(sweep)) / step_deg) + 1)
    t = np.linspace(a[0], a[-1], n)
    out = np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
    out[0], out[-1] = P[0], P[-1]
    return out


def has_arc(pts: np.ndarray, closed: bool) -> bool:
    """Vrai si la polyligne contient un arc ou un cercle net : à ne pas « redresser » en segments H/V (un cercle échantillonné
    à 4° a des segments à moins de 4° de l'horizontale ou de la verticale, qui seraient aplatis en losange)."""
    if len(pts) < 6:
        return False
    return any(pc[0] != "poly" for pc in fit_polyline(np.asarray(pts, float), closed, 0.0))
