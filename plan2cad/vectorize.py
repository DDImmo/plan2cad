"""Vectorisation par lignes centrales (centerline tracing) : squelette -> graphe -> polylignes.

Un mur plein de 15 px de large devient UN trait, les jonctions sont conservées
(les branches se rejoignent exactement au même point), les tronçons quasi
alignés sont fusionnés, puis simplifiés (Douglas-Peucker).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize

# voisinage circulaire : N, NE, E, SE, S, SW, W, NW
OFFS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
ORTHO_FIRST = [0, 2, 4, 6, 1, 3, 5, 7]


def skeleton_of(ink: np.ndarray) -> np.ndarray:
    return skeletonize(ink > 0)


def _components(nb, mask):
    n = len(mask)
    rows, cols = [], []
    for k in range(8):
        j = nb[:, k]
        ok = (j >= 0) & mask
        ok &= mask[np.where(j >= 0, j, 0)]
        rows.append(np.nonzero(ok)[0]); cols.append(j[ok])
    r = np.concatenate(rows); c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n)).tocsr()
    return connected_components(g, directed=False)[1]


def trace_skeleton(skel: np.ndarray, dt: np.ndarray | None = None):
    """Renvoie une liste de branches :
    dict(pts (N,2) x,y pixels, n0, n1 (id de jonction ou -1), closed, width_px)
    et le tableau des centroïdes de jonction."""
    ys, xs = np.nonzero(skel)
    n = len(ys)
    if n == 0:
        return [], np.zeros((0, 2))
    H, W = skel.shape
    idx = np.full((H + 2, W + 2), -1, np.int32)
    idx[ys + 1, xs + 1] = np.arange(n, dtype=np.int32)
    nb = np.empty((n, 8), np.int32)
    for k, (dy, dx) in enumerate(OFFS):
        nb[:, k] = idx[ys + 1 + dy, xs + 1 + dx]
    del idx
    has = nb >= 0
    trans = ((~has) & np.roll(has, -1, axis=1)).sum(1)
    junction = trans >= 3
    # Dilater les jonctions d'un pixel : sinon, en 8-connexité, les branches d'un « T »
    # restent reliées en diagonale une fois le pixel central retiré.
    near = np.zeros(n, bool)
    for k in range(8):
        j = nb[:, k]
        near |= (j >= 0) & junction[np.where(j >= 0, j, 0)]
    junction = junction | near
    nonj = ~junction

    # clusters de jonction
    jcent = np.zeros((0, 2))
    jlab = np.full(n, -1, np.int32)
    if junction.any():
        lab = _components(nb, junction)
        ids, inv = np.unique(lab[junction], return_inverse=True)
        jlab[junction] = inv
        cnt = np.bincount(inv)
        jcent = np.stack([np.bincount(inv, xs[junction] + 0.5) / cnt,
                          np.bincount(inv, ys[junction] + 0.5) / cnt], 1)

    # branches = composantes des pixels non-jonction
    blab = _components(nb, nonj)
    nodes = np.nonzero(nonj)[0]
    order = nodes[np.argsort(blab[nodes], kind="stable")]
    bl = blab[order]
    cuts = np.nonzero(np.diff(bl))[0] + 1
    groups = np.split(order, cuts)

    # degré dans le graphe non-jonction
    deg_nj = np.zeros(n, np.int8)
    for k in range(8):
        j = nb[:, k]
        deg_nj += ((j >= 0) & nonj[np.where(j >= 0, j, 0)]).astype(np.int8)

    nbl = nb.tolist()
    jl = junction.tolist()
    branches = []
    for g in groups:
        if len(g) == 0:
            continue
        ends = g[deg_nj[g] <= 1]
        start = int(ends[0]) if len(ends) else int(g[0])
        path = [start]
        seen = {start}
        cur = start
        while True:
            nxt = -1
            row = nbl[cur]
            for k in ORTHO_FIRST:
                m = row[k]
                if m >= 0 and not jl[m] and m not in seen:
                    nxt = m
                    break
            if nxt < 0:
                break
            path.append(nxt); seen.add(nxt); cur = nxt
        p = np.array(path)
        closed = False
        n0 = n1 = -1
        if len(ends) == 0 and len(p) >= 4:
            closed = any(nbl[int(p[-1])][k] == start for k in range(8))
        if not closed:
            def jn(node):
                out = []
                for k in range(8):
                    m = nbl[int(node)][k]
                    if m >= 0 and jl[m]:
                        c = int(jlab[m])
                        if c not in out:
                            out.append(c)
                return out
            a = jn(p[0]); b = jn(p[-1])
            if len(p) == 1:
                n0 = a[0] if a else -1
                n1 = a[1] if len(a) > 1 else -1
            else:
                n0 = a[0] if a else -1
                n1 = b[0] if b else -1
        pts = np.stack([xs[p] + 0.5, ys[p] + 0.5], 1)
        width = float(2 * np.median(dt[ys[p], xs[p]])) if dt is not None else 0.0
        branches.append(dict(pts=pts, n0=n0, n1=n1, closed=closed, width_px=width))
    return branches, jcent


def _dir_out(pts: np.ndarray, end: int, k: int = 10) -> np.ndarray:
    if end == 0:
        a, b = pts[0], pts[min(len(pts) - 1, k)]
    else:
        a, b = pts[-1], pts[max(0, len(pts) - 1 - k)]
    v = b - a
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def build_polylines(branches, jcent, prune_px: float, eps_px: float, min_len_px: float):
    """Élagage des épines, raccord aux jonctions, fusion des tronçons alignés, simplification.
    Renvoie [(pts_px (N,2), closed, width_px), ...]"""
    # 1. raccord exact aux centroïdes de jonction
    for b in branches:
        if b["closed"]:
            continue
        pts = b["pts"]
        if b["n0"] >= 0:
            pts = np.vstack([jcent[b["n0"]], pts])
        if b["n1"] >= 0:
            pts = np.vstack([pts, jcent[b["n1"]]])
        b["pts"] = pts
        b["len"] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
    for b in branches:
        if b["closed"]:
            b["len"] = float(np.linalg.norm(np.diff(b["pts"], axis=0), axis=1).sum())

    # 2. épines : tronçon court avec une extrémité libre et l'autre sur une jonction
    keep = []
    for b in branches:
        free0, free1 = b["n0"] < 0, b["n1"] < 0
        if not b["closed"] and (free0 != free1) and b["len"] < prune_px:
            continue
        keep.append(b)
    branches = keep

    # 3. simplification de chaque branche (extrémités = nœuds conservées)
    #    cercles et arcs nets : ajustés sur les pixels bruts (précision sous-pixel) avant toute simplification
    from .fit import refit_chain
    for b in branches:
        rc = refit_chain(b["pts"], b["closed"])
        if rc is not None:
            b["pts"] = rc
            b["fitted"] = True
    for b in branches:
        if b.get("fitted"):
            continue
        p = b["pts"].astype(np.float32).reshape(-1, 1, 2)
        if len(p) >= 3:
            s = cv2.approxPolyDP(p, eps_px, b["closed"]).reshape(-1, 2).astype(np.float64)
            b["pts"] = s if len(s) >= 2 else b["pts"]

    # 4. fusion aux jonctions
    ends_at: dict[int, list] = {}
    for i, b in enumerate(branches):
        if b["closed"]:
            continue
        if b["n0"] >= 0:
            ends_at.setdefault(b["n0"], []).append((i, 0))
        if b["n1"] >= 0:
            ends_at.setdefault(b["n1"], []).append((i, 1))
    link: dict[tuple, tuple] = {}
    for c, lst in ends_at.items():
        if len(lst) < 2:
            continue
        dirs = {e: _dir_out(branches[e[0]]["pts"], e[1]) for e in lst}
        if len(lst) == 2:
            a, b = lst
            if a[0] != b[0] and np.dot(dirs[a], dirs[b]) < 0.2:   # coude : on fusionne si pas rebroussement
                link[a] = b; link[b] = a
            continue
        cands = []
        for ia in range(len(lst)):
            for ib in range(ia + 1, len(lst)):
                a, b = lst[ia], lst[ib]
                if a[0] == b[0]:
                    continue
                d = float(np.dot(dirs[a], dirs[b]))
                if d < -0.9:
                    cands.append((d, a, b))
        for d, a, b in sorted(cands, key=lambda t: t[0]):
            if a in link or b in link:
                continue
            link[a] = b; link[b] = a

    used = set()
    out = []

    def oriented(i, start_end):
        p = branches[i]["pts"]
        return p if start_end == 0 else p[::-1]

    for i, b in enumerate(branches):
        if i in used:
            continue
        if b["closed"]:
            used.add(i)
            out.append((b["pts"], True, b["width_px"], b["len"]))
            continue
        # remonter jusqu'au début de chaîne
        cb, fe = i, 0
        guard = 0
        while (cb, fe) in link and guard < len(branches) + 2:
            b2, e2 = link[(cb, fe)]
            if b2 == i:
                break
            cb, fe = b2, 1 - e2
            guard += 1
        parts = [oriented(cb, fe)]
        wsum = branches[cb]["width_px"] * branches[cb]["len"]
        lsum = branches[cb]["len"]
        used.add(cb)
        exit_e = 1 - fe
        closed = False
        while (cb, exit_e) in link:
            b2, e2 = link[(cb, exit_e)]
            if b2 in used:
                closed = (b2 == i)
                break
            used.add(b2)
            parts.append(oriented(b2, e2)[1:])
            wsum += branches[b2]["width_px"] * branches[b2]["len"]
            lsum += branches[b2]["len"]
            cb, exit_e = b2, 1 - e2
        pts = np.vstack(parts)
        if closed and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        out.append((pts, closed, wsum / lsum if lsum else 0.0, lsum))

    # 5. filtre longueur
    res = []
    for pts, closed, w, ln in out:
        if len(pts) < 2:
            continue
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if L >= min_len_px:
            res.append((pts, closed, w))
    return res


# ------------------------------------------------------------------ murs en double trait
def detect_wall_gap(ink: np.ndarray, mm_per_px: float, lo_mm: float = 0.5, hi_mm: float = 4.0, n: int = 12):
    """Détecte des murs dessinés en DOUBLE TRAIT.

    Granulométrie du fond : si des « couloirs » blancs de largeur wall-scale sont très présents, ET si la
    majorité de la longueur des lignes (squelette) borde ces couloirs, alors les murs sont en double trait.
    Un simple tuyau parallèle à une cloison ne suffit pas.
    Renvoie (écart_mm_papier ou None, diagnostic)."""
    ds = max(1, int(0.1 / mm_per_px)) if mm_per_px < 0.1 else 1              # inutile de travailler sous ~0,1 mm/px
    work = ink if ds == 1 else cv2.resize(ink, (ink.shape[1] // ds, ink.shape[0] // ds), interpolation=cv2.INTER_AREA)
    work = (work > (0 if ds == 1 else 40)).astype(np.uint8)
    mpp = mm_per_px * ds
    bg = 1 - work
    ks = []
    for s in np.geomspace(lo_mm, hi_mm, n):
        k = max(1, int(round(s / mpp)))
        if not ks or k > ks[-1]:
            ks.append(k)
    opened = []
    for k in ks:
        opened.append(bg if k < 2 else cv2.morphologyEx(bg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)))
    areas = np.array([float(o.sum()) for o in opened])
    drop = -np.diff(areas)                                  # fond qui disparaît entre deux tailles
    if len(drop) < 4 or work.sum() <= 0:
        return None, dict(raison="image trop petite")
    i = int(np.argmax(drop))
    med = float(np.median(drop)) + 1e-9
    peak_ratio = float(drop[i] / med)
    last = i                                                # fin du pic dominant = épaisseur de mur principale
    gap_mm = float(ks[last + 1] * mpp)
    after = float(drop[i + 1:].max()) if i + 1 < len(drop) else 0.0
    diag = dict(ecart_mm=round(gap_mm, 2), pic_sur_mediane=round(min(peak_ratio, 999), 1),
                pic_interne=bool(i < len(drop) - 1 and after < 0.6 * drop[i]))
    if peak_ratio < 4.0 or not (lo_mm <= gap_mm <= hi_mm * 1.05) or not diag["pic_interne"]:
        diag["paires"] = None
        return None, diag
    # part de la longueur des lignes qui borde les couloirs de largeur « mur » (du plus fin significatif au pic)
    first = int(np.nonzero(drop >= 0.05 * drop[i])[0].min())
    ch = opened[first] & (1 - opened[last + 1])
    sk = skeletonize(work > 0)
    if not sk.any():
        return None, diag
    dist = cv2.distanceTransform((1 - ch).astype(np.uint8), cv2.DIST_L2, 3)
    near = float((dist[sk] <= max(3.0, 1.5 / mpp * 0.35 + 1.5)).mean())
    diag["paires"] = round(near, 2)
    if near >= 0.45:
        return gap_mm, diag
    return None, diag
