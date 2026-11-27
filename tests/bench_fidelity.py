"""Banc de fidélité : plan synthétique à géométrie CONNUE -> chaîne complète -> écart au vrai tracé.

Le plan (murs épais H/V, mur oblique à 30°, mur courbe, deux portes avec leur arc de battant, un poteau rond,
une cote) est dessiné en suréchantillonnage puis réduit : les bords sont donc anti-aliasés à des positions
sous-pixel, comme sur un vrai scan. Ensuite bruit, flou, poussières, ombrage.

Mesures (mm papier, 400 dpi => 1 px = 0,0635 mm) :
  rappel     : distance de chaque point du vrai tracé au tracé produit
  precision  : distance de chaque point produit au vrai tracé
  entites    : nombre de LINE / POLYLINE / ARC / CIRCLE écrits dans le DXF et nombre de sommets

Usage : python tests/bench_fidelity.py [dossier_de_sortie]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan2cad.demo import _scan_effects                      # noqa: E402
from plan2cad.dxf import read_dxf                            # noqa: E402
from plan2cad.pipeline import Options, run_project           # noqa: E402

W, H = 3200, 2200
DPI = 400.0
MM_PX = 25.4 / DPI
SS = 4                                                       # suréchantillonnage du dessin


# ------------------------------------------------------------------ vérité terrain
def scene():
    """Primitives en px (axe du tracé) : ('line', p0, p1, épaisseur) / ('arc', c, r, a0, a1, épaisseur) / ('circle', c, r, ép.).
    Angles en degrés, sens horaire à l'écran (convention cv2.ellipse, y vers le bas)."""
    t_out, t_in, thin = 40, 24, 4
    P = []
    x0, y0, x1, y1 = 300, 300, 2900, 1900
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        P.append(("line", a, b, t_out))
    P.append(("line", (1200, 300), (1200, 1000), t_in))
    P.append(("line", (1200, 1130), (1200, 1900), t_in))
    P.append(("line", (1200, 1200), (1900, 1200), t_in))
    P.append(("line", (2030, 1200), (2900, 1200), t_in))
    c30, s30 = np.cos(np.radians(30)), np.sin(np.radians(30))
    P.append(("line", (2100, 1900), (2100 + 700 * c30, 1900 - 700 * s30), t_in))          # oblique 30°
    P.append(("arc", (700, 1500), 300, 200, 320, t_in))                                    # mur courbe
    P.append(("arc", (1200, 1130), 130, 270, 360, thin))                                   # battant porte 1
    P.append(("line", (1200, 1130), (1330, 1130), thin))
    P.append(("arc", (2030, 1200), 130, 180, 270, thin))                                   # battant porte 2
    P.append(("line", (2030, 1200), (2030, 1070), thin))
    P.append(("circle", (1700, 700), 45, 6))                                               # poteau rond
    P.append(("line", (300, 2050), (2900, 2050), 3))                                       # cote
    P.append(("line", (300, 2020), (300, 2080), 3))
    P.append(("line", (2900, 2020), (2900, 2080), 3))
    return P


def render(prims, seed=7, angle=0.0):
    big = np.full((H * SS, W * SS), 255, np.uint8)
    ink = 25

    def pt(p):
        return (int(round(p[0] * SS)), int(round(p[1] * SS)))

    for pr in prims:
        if pr[0] == "line":
            cv2.line(big, pt(pr[1]), pt(pr[2]), ink, max(1, int(round(pr[3] * SS))), cv2.LINE_8)
        elif pr[0] == "arc":
            c, r, a0, a1, th = pr[1:]
            cv2.ellipse(big, pt(c), (int(r * SS), int(r * SS)), 0, a0, a1, ink, max(1, int(round(th * SS))), cv2.LINE_8)
        elif pr[0] == "circle":
            cv2.circle(big, pt(pr[1]), int(pr[2] * SS), ink, max(1, int(round(pr[3] * SS))), cv2.LINE_8)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    img, M = _scan_effects(img, rng, angle, 1.0, 0, 0)
    return img, M


def sample_truth(prims, step=2.0):
    out = []
    for pr in prims:
        if pr[0] == "line":
            a, b = np.asarray(pr[1], float), np.asarray(pr[2], float)
            n = max(2, int(np.hypot(*(b - a)) / step))
            t = np.linspace(0, 1, n)[:, None]
            out.append(a + (b - a) * t)
        elif pr[0] in ("arc", "circle"):
            c = np.asarray(pr[1], float)
            r = pr[2]
            a0, a1 = (pr[3], pr[4]) if pr[0] == "arc" else (0.0, 360.0)
            n = max(8, int(np.radians(a1 - a0) * r / step))
            ang = np.radians(np.linspace(a0, a1, n))
            out.append(np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
    return np.vstack(out)


# ------------------------------------------------------------------ mesure
def densify(polys, step=2.0):
    out = []
    for pts, closed in polys:
        p = np.asarray(pts, float)
        if closed:
            p = np.vstack([p, p[:1]])
        for a, b in zip(p[:-1], p[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step) + 1)
            t = np.linspace(0, 1, n)[:, None]
            out.append(a + (b - a) * t)
    return np.vstack(out) if out else np.zeros((0, 2))


def dxf_stats(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    kinds = {}
    for i in range(0, len(lines) - 1, 2):
        if lines[i].strip() == "0":
            v = lines[i + 1].strip()
            kinds[v] = kinds.get(v, 0) + 1
    return {k: kinds.get(k, 0) for k in ("LINE", "POLYLINE", "LWPOLYLINE", "ARC", "CIRCLE", "VERTEX")}


def evaluate(dxf_path, truth_px, scale=100.0, skip_layers=("Z-POINTS-ANCRAGE",)):
    ents, _, _ = read_dxf(dxf_path)
    k = MM_PX * scale
    polys = []
    for e in ents:
        if e["type"] != "poly" or e["layer"] in skip_layers:
            continue
        p = e["pts"]
        polys.append((np.c_[p[:, 0] / k, H - p[:, 1] / k], e["closed"]))
    out = densify(polys)
    tr_out, tr_gt = cKDTree(out), cKDTree(truth_px)
    d_rec = tr_out.query(truth_px)[0] * MM_PX
    d_pre = tr_gt.query(out)[0] * MM_PX
    return dict(
        rappel_moy_mm=float(d_rec.mean()), rappel_p95_mm=float(np.percentile(d_rec, 95)), rappel_max_mm=float(d_rec.max()),
        precision_moy_mm=float(d_pre.mean()), precision_p95_mm=float(np.percentile(d_pre, 95)),
        precision_max_mm=float(d_pre.max()),
        part_trace_retrouve_a_0p10mm=float((d_rec <= 0.10).mean()), part_trace_retrouve_a_0p25mm=float((d_rec <= 0.25).mean()),
    ), polys


def overlay(img, truth_px, polys, path):
    big = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    big = (255 - (255 - big) // 3).astype(np.uint8)
    for p in truth_px[::3]:
        cv2.circle(big, (int(p[0]), int(p[1])), 1, (60, 170, 60), -1)
    for pts, closed in polys:
        cv2.polylines(big, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], closed, (40, 40, 220), 1, cv2.LINE_AA)
        for q in pts:
            cv2.circle(big, (int(round(q[0])), int(round(q[1]))), 3, (200, 0, 0), -1)
    cv2.imwrite(str(path), big)


def main(outdir: str | None = None, label: str = "run", **opt_kw):
    out = Path(outdir or "bench_out") / label
    out.mkdir(parents=True, exist_ok=True)
    prims = scene()
    img, _ = render(prims)
    src = out / "plan_synthetique.png"
    cv2.imwrite(str(src), img)
    truth_px = sample_truth(prims)
    opts = Options(dpi=DPI, scale=100.0, ocr=False, max_px=max(W, H), **opt_kw)
    project = dict(name="bench", levels=[dict(name="RDC", files=dict(arc=str(src)))])
    run_project(project, opts, str(out / "resultat"), echo=False)
    dxf = out / "resultat" / "bench.dxf"
    metr, polys = evaluate(dxf, truth_px)
    metr["entites_dxf"] = dxf_stats(dxf)
    overlay(img, truth_px, polys, out / "comparaison.png")
    (out / "metriques.json").write_text(json.dumps(metr, indent=1, ensure_ascii=False), encoding="utf-8")
    return metr


if __name__ == "__main__":
    m = main(sys.argv[1] if len(sys.argv) > 1 else None, label=sys.argv[2] if len(sys.argv) > 2 else "run")
    print(json.dumps(m, indent=1, ensure_ascii=False))
