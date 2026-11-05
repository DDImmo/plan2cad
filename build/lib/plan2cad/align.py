"""Calage automatique : remplace la commande ALIGNER faite à la main.

Deux modes :
  * ancres : deux points repères (pixels du scan -> pixels du plan maître) => similarité exacte ;
  * auto   : Fourier-Mellin (rotation + échelle) + corrélation de phase (translation),
             puis affinage par ICP à similarité. Le meilleur candidat est choisi par distance de chanfrein.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .model import (Poly, Sim, IDENTITY, sim_apply, sim_compose, sim_from_pairs,
                    bbox, sample_points)

N_CANVAS = 1024


def _union_canvas(a: list[Poly], b: list[Poly], margin=0.05):
    ba, bb = bbox(a), bbox(b)
    x0 = min(ba[0], bb[0]); y0 = min(ba[1], bb[1])
    x1 = max(ba[2], bb[2]); y1 = max(ba[3], bb[3])
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - side / 2, cy - side / 2, side


def _raster(polys: list[Poly], ox, oy, side, n=N_CANVAS):
    img = np.zeros((n, n), np.uint8)
    k = n / side
    for pl in polys:
        p = np.round((pl.pts - [ox, oy]) * k).astype(np.int32)
        cv2.polylines(img, [p.reshape(-1, 1, 2)], pl.closed, 255, 1)
    return img


def _logpolar(img):
    n = img.shape[0]
    win = cv2.createHanningWindow((n, n), cv2.CV_32F)
    f = np.abs(np.fft.fftshift(np.fft.fft2(img.astype(np.float32) * win)))
    f = np.log1p(f).astype(np.float32)
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    f *= np.clip((r - 0.04) / 0.2, 0, 1).astype(np.float32)        # passe-haut
    maxr = n / 2
    lp = cv2.warpPolar(f, (n, n), (n / 2, n / 2), maxr, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    M = n / np.log(maxr)
    return lp, M


def _sim_about(center: complex, scale: float, rot_rad: float, shift: complex = 0j) -> Sim:
    p = scale * np.exp(1j * rot_rad)
    return (p, center - p * center + shift)


def _chamfer(dist_img, pts_world, ox, oy, side, cap=30.0):
    k = dist_img.shape[0] / side
    uv = (pts_world - [ox, oy]) * k
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, dist_img.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, dist_img.shape[0] - 1)
    d = dist_img[v, u]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < dist_img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < dist_img.shape[0])
    d = np.where(inside, d, cap)
    return float(np.minimum(d, cap).mean())


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Sim:
    a = src[:, 0] + 1j * src[:, 1]
    b = dst[:, 0] + 1j * dst[:, 1]
    ma, mb = a.mean(), b.mean()
    A, B = a - ma, b - mb
    den = (np.abs(A) ** 2).sum()
    if den <= 0:
        return IDENTITY
    p = (np.conj(A) * B).sum() / den
    return (p, mb - p * ma)


def icp(master_pts, moving_pts, sim0: Sim, iters=60, keep=0.7):
    tree = cKDTree(master_pts)
    sim = sim0
    for _ in range(iters):
        cur = sim_apply(sim, moving_pts)
        d, i = tree.query(cur)
        thr = np.quantile(d, keep)
        m = d <= max(thr, 1e-9)
        if m.sum() < 3:
            break
        step = _umeyama(cur[m], master_pts[i[m]])
        new = sim_compose(sim, step)
        if abs(step[0] - 1) < 1e-7 and abs(step[1]) < 1e-4:
            sim = new
            break
        sim = new
    return sim


def auto_align(master: list[Poly], moving: list[Poly], log=print):
    """Renvoie (sim, métriques). sim envoie `moving` sur `master`."""
    ox, oy, side = _union_canvas(master, moving)
    n = N_CANVAS
    k = n / side
    img_m = _raster(master, ox, oy, side)
    img_v = _raster(moving, ox, oy, side)
    dist = cv2.distanceTransform(255 - img_m, cv2.DIST_L2, 3)
    diag = float(np.hypot(*(np.array(bbox(master)[2:]) - np.array(bbox(master)[:2]))))
    step = diag / 500
    mp = sample_points(master, step)
    vp = sample_points(moving, step)
    center = complex(ox + side / 2, oy + side / 2)

    lpm, M = _logpolar(img_m)
    lpv, _ = _logpolar(img_v)
    (dx, dy), _resp = cv2.phaseCorrelate(lpm.astype(np.float32), lpv.astype(np.float32))
    scale0 = float(np.exp(dx / M))
    ang0 = dy * 360.0 / n

    cands = []
    for ss in (1, -1):
        for sa in (1, -1):
            for add in (0.0, 180.0):
                s = scale0 ** ss
                a = np.radians(sa * ang0 + add)
                if 0.2 < s < 5:
                    cands.append((s, a))
    cands.append((1.0, 0.0))
    # candidat "boîte englobante" : même emprise
    bm, bv = bbox(master), bbox(moving)
    sb = np.hypot(bm[2] - bm[0], bm[3] - bm[1]) / max(np.hypot(bv[2] - bv[0], bv[3] - bv[1]), 1e-9)
    cands.append((sb, 0.0))

    best = None
    for s, a in cands:
        s0 = _sim_about(center, s, a)
        # translation par corrélation de phase sur le raster déplacé
        A = np.array([[s * np.cos(a), -s * np.sin(a)], [s * np.sin(a), s * np.cos(a)]])
        c_px = np.array([(center.real - ox) * k, (center.imag - oy) * k])
        t = c_px - A @ c_px
        warped = cv2.warpAffine(img_v, np.hstack([A, t[:, None]]).astype(np.float64), (n, n))
        (tx, ty), resp = cv2.phaseCorrelate(img_m.astype(np.float32), warped.astype(np.float32))
        for sg in (1, -1):
            shift = complex(sg * tx / k, sg * ty / k)
            cand = _sim_about(center, s, a, shift)
            sc = _chamfer(dist, sim_apply(cand, vp), ox, oy, side)
            if best is None or sc < best[0]:
                best = (sc, cand)
        # candidat sans décalage (bbox centres) pour robustesse
        cvc = complex(*(np.array([bm[0] + bm[2], bm[1] + bm[3]]) / 2))
        cmv = sim_apply(s0, np.array([[(bv[0] + bv[2]) / 2, (bv[1] + bv[3]) / 2]]))[0]
        cand = _sim_about(center, s, a, cvc - complex(*cmv))
        sc = _chamfer(dist, sim_apply(cand, vp), ox, oy, side)
        if sc < best[0]:
            best = (sc, cand)

    sim = icp(mp, vp, best[1])
    # second passage ICP plus strict
    sim = icp(mp, vp, sim, iters=40, keep=0.55)
    tree = cKDTree(mp)
    aligned = sim_apply(sim, vp)
    d, _ = tree.query(aligned)
    tol = diag * 0.004
    d_back, _ = cKDTree(aligned).query(mp)             # part du plan maître retrouvée dans le plan calé
    metrics = dict(
        coarse_chamfer_px=round(best[0], 2),
        inlier_ratio=round(float((d < tol).mean()), 3),
        coverage=round(float((d_back < tol).mean()), 3),
        median_residual_mm=round(float(np.median(d)), 1),
        tolerance_mm=round(tol, 1),
    )
    return sim, metrics


def anchor_align(src_world: np.ndarray, dst_world: np.ndarray):
    """Similarité exacte à partir de 2 paires de points (déjà en coordonnées monde)."""
    sim = sim_from_pairs(np.asarray(src_world, float), np.asarray(dst_world, float))
    return sim


def overlap_ratio(polys: list[Poly], master: list[Poly], tol: float) -> float:
    """Part (en longueur échantillonnée) de `polys` qui recouvre la géométrie du maître à `tol` près."""
    mp = sample_points(master, tol / 2, max_pts=400000)
    vp = sample_points(polys, tol / 2, max_pts=200000)
    if len(mp) == 0 or len(vp) == 0:
        return 0.0
    d, _ = cKDTree(mp).query(vp)
    return float((d <= tol).mean())


def strip_backdrop(polys: list[Poly], master: list[Poly], tol: float):
    """Retire des polylignes les portions qui recouvrent le plan maître (fond d'architecture
    présent sur les plans de lots). Renvoie (polylignes, part_retirée_en_longueur)."""
    mp = sample_points(master, tol / 2, max_pts=400000)
    tree = cKDTree(mp)
    out: list[Poly] = []
    total = removed = 0.0
    step = tol / 2
    for pl in polys:
        p = pl.pts
        if pl.closed and len(p) > 2:
            p = np.vstack([p, p[:1]])
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        L = float(seg.sum())
        total += L
        if L <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(seg)])
        t = np.linspace(0, L, max(2, int(L / step) + 2))
        dense = np.stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])], 1)
        far = tree.query(dense)[0] > tol
        if far.all():
            out.append(pl)
            continue
        if not far.any():
            removed += L
            continue
        f = np.concatenate([[False], far, [False]]).astype(np.int8)
        d = np.diff(f)
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        kept = 0.0
        for a, b in zip(starts, ends):
            a0, b0 = max(0, a - 1), min(len(dense) - 1, b + 1)
            sub = dense[a0:b0 + 1]
            ln = float(np.linalg.norm(np.diff(sub, axis=0), axis=1).sum()) if len(sub) > 1 else 0.0
            if ln < 2 * tol:
                continue
            simp = cv2.approxPolyDP(sub.astype(np.float32).reshape(-1, 1, 2), tol / 3, False).reshape(-1, 2)
            out.append(Poly(simp.astype(np.float64), pl.layer, False, pl.width, pl.disc))
            kept += ln
        removed += max(0.0, L - kept)
    return out, (removed / total if total else 0.0)


def footprint_iou(a: list[Poly], b: list[Poly], n: int = 512) -> float:
    """Recouvrement (IoU) des emprises remplies de deux jeux de polylignes déjà dans le même repère."""
    ox, oy, side = _union_canvas(a, b)
    out = []
    for polys in (a, b):
        img = _raster(polys, ox, oy, side, n)
        img = cv2.dilate(img, np.ones((5, 5), np.uint8))
        flood = img.copy()
        mask = np.zeros((n + 2, n + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 128)
        out.append(flood != 128)                       # tout ce qui n'est pas le fond extérieur
    inter = float((out[0] & out[1]).sum())
    union = float((out[0] | out[1]).sum())
    return inter / union if union else 0.0
