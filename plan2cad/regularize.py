"""Nettoyage géométrique des tracés issus d'un SCAN : orientation et régularisation.

Le squelette d'un scan « tremble » (quelques mm d'écart sur un mur qui devrait être droit) et la feuille est rarement posée
bien droite. Deux corrections, appliquées aux murs d'un plan d'architecture :
  1. redressement : l'angle dominant des segments (modulo 90°) est ramené à 0° ;
  2. régularisation : les suites de segments quasi horizontaux / verticaux deviennent UN segment exact, les murs qui
     devraient être alignés partagent la même coordonnée, les angles sont recalculés par intersection.
Les tracés obliques ou courbes (arcs de porte, façades, escaliers) ne sont pas redressés : seuls leurs points de raccord
avec un mur droit sont déplacés.
"""
from __future__ import annotations

import numpy as np

AX_TOL_DEG = 4.0


def _seg_info(pts: np.ndarray):
    d = np.diff(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return d, L, ang


def dominant_skew(polys: list, min_share: float = 0.25):
    """Angle (°) à retirer pour que les murs soient horizontaux / verticaux, et part des longueurs concernées.
    Renvoie (theta, share) ou (0.0, 0.0) si aucune orientation dominante."""
    angs, ws = [], []
    for p in polys:
        if len(p.pts) < 2:
            continue
        pts = p.pts if not p.closed else np.vstack([p.pts, p.pts[:1]])
        _, L, ang = _seg_info(pts)
        keep = L >= 0.0
        angs.append(ang[keep]); ws.append(L[keep])
    if not angs:
        return 0.0, 0.0
    a = np.concatenate(angs); w = np.concatenate(ws)
    if w.sum() <= 0:
        return 0.0, 0.0
    m = ((a + 45.0) % 90.0) - 45.0                                   # modulo 90° dans [-45, 45)
    bins = np.arange(-45.0, 45.0 + 0.1, 0.1)
    h, _ = np.histogram(m, bins=bins, weights=w)
    k = np.ones(11) / 11.0                                            # lissage ±0,5°
    hs = np.convolve(np.r_[h[-5:], h, h[:5]], k, mode="same")[5:-5]
    i = int(np.argmax(hs))
    theta = float((bins[i] + bins[i + 1]) / 2)
    near = np.abs(((m - theta + 45.0) % 90.0) - 45.0) <= 1.5
    share = float(w[near].sum() / w.sum())
    if share < min_share:
        return 0.0, share
    # affinage : moyenne pondérée des angles proches du pic
    sel = np.abs(((m - theta + 45.0) % 90.0) - 45.0) <= 1.5
    theta = float(np.average(((m[sel] - theta + 45.0) % 90.0) - 45.0 + theta, weights=w[sel]))
    return theta, share


def rotate(polys: list, texts: list, theta_deg: float, center=None):
    """Tourne polylignes et textes de -theta autour de `center` (par défaut le centre de l'emprise)."""
    from .model import Poly, Text
    if not polys:
        return polys, texts
    if center is None:
        allp = np.vstack([p.pts for p in polys])
        center = (allp.min(0) + allp.max(0)) / 2.0
    t = np.radians(-theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.asarray(center, float)
    out = [Poly((p.pts - c) @ R.T + c, p.layer, p.closed, p.width, p.disc) for p in polys]
    tout = []
    for x in texts or []:
        pos = (np.asarray(x.pos, float) - c) @ R.T + c
        tout.append(Text(pos, x.s, x.height, x.layer, (x.angle - theta_deg) % 360.0, x.conf))
    return out, tout


# ------------------------------------------------------------------ régularisation
def _runs(pts: np.ndarray, tol_dev: float):
    """Découpe une polyligne en 'runs' : suites de segments quasi H ou V (un seul segment exact) ou segments libres."""
    d, L, ang = _seg_info(pts)
    # h : angle proche de 0/180 ; v : proche de ±90
    kind = []
    for a in ang:
        aa = abs(a)
        if aa <= AX_TOL_DEG or aa >= 180.0 - AX_TOL_DEG:
            kind.append("h")
        elif abs(aa - 90.0) <= AX_TOL_DEG:
            kind.append("v")
        else:
            kind.append("o")
    runs = []
    i = 0
    n = len(L)
    while i < n:
        k = kind[i]
        if k == "o":
            runs.append(dict(kind="o", i0=i, i1=i + 1))
            i += 1
            continue
        j = i
        ax = 1 if k == "h" else 0                                    # coordonnée constante (h : y ; v : x)
        while j + 1 < n and kind[j + 1] == k:
            seg_pts = pts[i:j + 3]
            w = np.r_[L[i:j + 2]]
            mid = np.average((seg_pts[:-1, ax] + seg_pts[1:, ax]) / 2, weights=np.maximum(w, 1e-9))
            if np.abs(seg_pts[:, ax] - mid).max() > tol_dev:
                break
            j += 1
        sp = pts[i:j + 2]
        w = L[i:j + 1]
        c = float(np.average((sp[:-1, ax] + sp[1:, ax]) / 2, weights=np.maximum(w, 1e-9)))
        runs.append(dict(kind=k, i0=i, i1=j + 1, c=c, w=float(w.sum())))
        i = j + 1
    return runs


def _cluster_snap(items: list, tol: float):
    """items : liste de dict(c, w) ; renvoie pour chacun la coordonnée partagée (moyenne pondérée du groupe)."""
    if not items:
        return
    order = sorted(range(len(items)), key=lambda i: items[i]["c"])
    group = [order[0]]
    groups = []
    for i in order[1:]:
        if items[i]["c"] - items[group[-1]]["c"] <= tol:
            group.append(i)
        else:
            groups.append(group)
            group = [i]
    groups.append(group)
    for g in groups:
        w = np.array([items[i]["w"] for i in g])
        c = float(np.average([items[i]["c"] for i in g], weights=np.maximum(w, 1e-9)))
        for i in g:
            items[i]["snap"] = c


def _drop_collinear(pts: np.ndarray, closed: bool, deg: float = 0.8):
    if len(pts) < 3:
        return pts
    keep = np.ones(len(pts), bool)
    n = len(pts)
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        n1, n2 = np.hypot(*v1), np.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            keep[i] = False
            continue
        cos = np.clip((v1 @ v2) / (n1 * n2), -1, 1)
        if np.degrees(np.arccos(cos)) < deg:
            keep[i] = False
    out = pts[keep]
    return out if len(out) >= 2 else pts


def regularize(polys: list, scale: float, layers=("ARC-MURS-INT", "ARC-MURS-EXT"), tol_dev_mm: float = 0.9,
               tol_snap_mm: float = 0.35):
    """Régularise les murs. `scale` : mm réels par mm de papier (les tolérances sont exprimées en mm de papier).
    Renvoie (polys, stats)."""
    from .model import Poly
    tol_dev = tol_dev_mm * scale                       # un mur « droit » peut dévier de 0,9 mm de papier
    tol_snap = tol_snap_mm * scale                     # deux murs à moins de 0,35 mm de papier sont alignés
    short = 2.5 * tol_dev
    info = []                                    # un dict par polyligne traitée
    all_h, all_v = [], []
    for pi, p in enumerate(polys):
        if p.layer not in layers or len(p.pts) < 2:
            continue
        pts = p.pts if not p.closed else np.vstack([p.pts, p.pts[:1]])
        runs = _runs(pts, tol_dev)
        # petits segments obliques coincés entre un H et un V : coin arrondi par le squelette -> supprimés (intersection)
        for r in runs:
            if r["kind"] != "o":
                (all_h if r["kind"] == "h" else all_v).append(r)
        info.append((pi, pts, runs))
    _cluster_snap(all_h, tol_snap)
    _cluster_snap(all_v, tol_snap)
    out = list(polys)
    merged = 0
    for pi, pts, runs in info:
        p = polys[pi]
        # supprimer les petits 'o' entre deux runs d'orientations différentes
        cleaned = []
        for j, r in enumerate(runs):
            if (r["kind"] == "o" and 0 < j < len(runs) - 1 and runs[j - 1]["kind"] in "hv" and runs[j + 1]["kind"] in "hv"
                    and runs[j - 1]["kind"] != runs[j + 1]["kind"]
                    and np.hypot(*(pts[r["i1"]] - pts[r["i0"]])) <= short):
                continue
            cleaned.append(r)
        runs = cleaned
        newpts = []
        for j, r in enumerate(runs):
            a = pts[r["i0"]].copy()
            b = pts[r["i1"]].copy()
            if r["kind"] == "h":
                a[1] = b[1] = r["snap"]
            elif r["kind"] == "v":
                a[0] = b[0] = r["snap"]
            if j > 0:
                prev = runs[j - 1]
                if prev["kind"] in "hv" and r["kind"] in "hv" and prev["kind"] != r["kind"]:
                    hh, vv = (prev, r) if prev["kind"] == "h" else (r, prev)
                    corner = np.array([vv["snap"], hh["snap"]])
                    newpts[-1] = corner
                    a = corner
            if j == 0:
                newpts.append(a)
            elif not np.allclose(newpts[-1], a):
                newpts.append(a)
            newpts.append(b)
        if p.closed and len(runs) >= 2 and runs[0]["kind"] in "hv" and runs[-1]["kind"] in "hv" \
                and runs[0]["kind"] != runs[-1]["kind"]:
            hh, vv = (runs[0], runs[-1]) if runs[0]["kind"] == "h" else (runs[-1], runs[0])
            corner = np.array([vv["snap"], hh["snap"]])
            newpts[0] = corner
            newpts[-1] = corner
        arr = np.array(newpts, float)
        if p.closed and len(arr) > 2:
            if np.allclose(arr[0], arr[-1]):
                arr = arr[:-1]
            else:
                # raccord de fermeture : si premier et dernier run sont de même nature, le point de départ est déjà exact
                arr = arr[:-1] if np.hypot(*(arr[0] - arr[-1])) <= short else arr
        arr = _drop_collinear(arr, p.closed)
        merged += max(0, len(p.pts) - len(arr))
        if len(arr) >= 2:
            out[pi] = Poly(arr, p.layer, p.closed, p.width, p.disc)
    return out, dict(polylignes=len(info), sommets_supprimes=merged,
                     segments_alignes=len(all_h) + len(all_v))


def axis_exactness(polys: list, layers=("ARC-MURS-INT", "ARC-MURS-EXT"), tol_deg: float = 0.05):
    """Part de la longueur des murs QUASI H/V (≤ 4°) qui est EXACTEMENT H/V (≤ tol_deg) : mesure de la régularité."""
    tot = ex = 0.0
    for p in polys:
        if p.layer not in layers or len(p.pts) < 2:
            continue
        pts = p.pts if not p.closed else np.vstack([p.pts, p.pts[:1]])
        _, L, ang = _seg_info(pts)
        for l, a in zip(L, ang):
            r = ((a + 45.0) % 90.0) - 45.0
            if abs(r) <= AX_TOL_DEG:
                tot += l
                if abs(r) <= tol_deg:
                    ex += l
    return ex / tot if tot else 0.0


# ------------------------------------------------------------------ obliques, arcs, PDF vectoriels
def _fit_circle(P: np.ndarray):
    """Cercle des moindres carrés (algébrique) : (cx, cy, r, résidu rms)."""
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(P))]
    b = x * x + y * y
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def _arc_points(P: np.ndarray, c, step_deg: float = 6.0):
    """Arc propre de même départ / arrivée que P, échantillonné régulièrement sur le cercle ajusté."""
    cx, cy, r, _ = c
    a0 = np.arctan2(P[0, 1] - cy, P[0, 0] - cx)
    a1 = np.arctan2(P[-1, 1] - cy, P[-1, 0] - cx)
    mid = P[len(P) // 2]
    am = np.arctan2(mid[1] - cy, mid[0] - cx)
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi                        # sens le plus court a0 -> a1
    dm = (am - a0 + np.pi) % (2 * np.pi) - np.pi
    if d * dm < 0 or abs(dm) > abs(d):                                  # l'arc passe par l'autre côté
        d = d - np.sign(d) * 2 * np.pi if d != 0 else 2 * np.pi
    n = max(3, int(abs(np.degrees(d)) / step_deg) + 1)
    ang = a0 + np.linspace(0, d, n)
    out = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    out[0], out[-1] = P[0], P[-1]
    return out, abs(np.degrees(d))


def regularize_oblique(polys: list, scale: float, layers=("ARC-MURS-INT", "ARC-MURS-EXT", "ARC-PORTE"),
                       dev_mm_paper: float = 0.9, snap_deg: float = 1.5):
    """Obliques et courbes : (1) une suite de segments obliques alignés devient UN segment ; (2) une suite qui suit un
    cercle devient un arc net (arcs de porte, murs arrondis) ; (3) les angles obliques proches de multiples de 15°
    (30, 45, 60...) ou d'une orientation partagée par plusieurs façades sont alignés. Les raccords avec les murs droits
    sont conservés. Renvoie (polys, stats)."""
    from .model import Poly
    tol = dev_mm_paper * scale
    out = list(polys)
    chains = []                                                           # (poly index, i0, i1, kind, data)
    n_line = n_arc = 0
    for pi, p in enumerate(polys):
        if p.layer not in layers or len(p.pts) < 3:
            continue
        pts = p.pts
        _, L, ang = _seg_info(pts)
        obl = []
        for a in ang:
            aa = abs(a)
            obl.append(not (aa <= AX_TOL_DEG or aa >= 180 - AX_TOL_DEG or abs(aa - 90) <= AX_TOL_DEG))
        i = 0
        n = len(L)
        edits = []
        while i < n:
            if not obl[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and obl[j + 1]:
                j += 1
            P = pts[i:j + 2]
            chord = P[-1] - P[0]
            cl = float(np.hypot(*chord))
            if cl > 1e-6 and len(P) >= 2:
                u = chord / cl
                dev = np.abs((P[:, 0] - P[0, 0]) * u[1] - (P[:, 1] - P[0, 1]) * u[0])
                if dev.max() <= tol:
                    edits.append((i, j + 1, "line", P[[0, -1]]))
                elif len(P) >= 4 and cl > 3 * tol:
                    c = _fit_circle(P)
                    if c and 4 * tol <= c[2] <= 4000 and c[3] <= 0.03 * c[2] and dev.max() >= 0.06 * cl:
                        arc, sweep = _arc_points(P, c)
                        if 25 <= sweep <= 200:
                            edits.append((i, j + 1, "arc", arc))
            i = j + 1
        if not edits:
            continue
        newp = pts
        for i0, i1, kind, rep in sorted(edits, key=lambda e: -e[0]):
            newp = np.vstack([newp[:i0], rep, newp[i1 + 1:]])
            n_line += kind == "line"
            n_arc += kind == "arc"
        first_pt_idx = 0
        for i0, i1, kind, rep in sorted(edits, key=lambda e: e[0]):
            if kind == "line":
                chains.append((pi, i0, i1))
        out[pi] = Poly(newp, p.layer, p.closed, p.width, p.disc)
    # alignement des angles des obliques droites (poly entièrement oblique et ouverte, ou extrémité libre)
    segs = []
    for pi, p in enumerate(out):
        if p.layer not in layers or p.closed or len(p.pts) != 2:
            continue
        d = p.pts[1] - p.pts[0]
        L = float(np.hypot(*d))
        a = np.degrees(np.arctan2(d[1], d[0])) % 180.0
        if min(a % 90, 90 - a % 90) > AX_TOL_DEG:
            segs.append((pi, a, L))
    snapped = 0
    if segs:
        arr = sorted(segs, key=lambda s: s[1])
        groups, cur = [], [arr[0]]
        for s in arr[1:]:
            if s[1] - cur[-1][1] <= 1.2:
                cur.append(s)
            else:
                groups.append(cur); cur = [s]
        groups.append(cur)
        for g in groups:
            m = float(np.average([s[1] for s in g], weights=[s[2] for s in g]))
            near15 = round(m / 15.0) * 15.0
            target = near15 if abs(m - near15) <= snap_deg else (m if len(g) >= 3 else None)
            if target is None:
                continue
            for pi, a, L in g:
                p = out[pi]
                c = p.pts.mean(0)
                t = np.radians(target)
                v = np.array([np.cos(t), np.sin(t)]) * L / 2
                if np.dot(v, p.pts[1] - p.pts[0]) < 0:
                    v = -v
                new = np.array([c - v, c + v])
                if np.abs(new - p.pts).max() <= 2.0 * tol:
                    out[pi] = Poly(new, p.layer, p.closed, p.width, p.disc)
                    snapped += 1
    return out, dict(obliques_droites=n_line, arcs_nets=n_arc, obliques_alignees=snapped)


def rotate_pdf(paths: list, texts: list, theta_up: float, center):
    """Tourne les tracés et textes d'un PDF vectoriel (coordonnées PDF, y vers le bas) de -theta (theta mesuré y vers le
    haut, comme `dominant_skew` sur le monde). Renvoie de nouvelles listes (le cache de lecture n'est pas modifié)."""
    t = np.radians(theta_up)                       # y vers le bas : la rotation inverse de theta_up devient +theta_up
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.asarray(center, float)

    def rot(a):
        return (np.asarray(a, float) - c) @ R.T + c
    np_ = []
    for p in paths:
        q = rot(p["pts"])
        np_.append(dict(p, pts=q, bbox=[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]))
    nt = []
    for s in texts:
        b = s["bb"]
        corners = rot([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        o = rot([[s["x"], s["y"]]])[0]
        d = dict(s, x=float(o[0]), y=float(o[1]), angle=(s["angle"] - theta_up) % 360.0,
                 bb=[corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
        if "end" in s and s["end"] is not None:
            e = rot([list(s["end"])])[0]
            d["end"] = (float(e[0]), float(e[1]))
        nt.append(d)
    return np_, nt
