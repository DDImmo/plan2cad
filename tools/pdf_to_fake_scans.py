"""PDF vectoriel -> faux scans (un par plan détecté) : pour tester le mode scan brut avec une vérité terrain.

Chaque plan de la planche est rastérisé (défaut 300 dpi) puis reçoit les défauts d'un scan : fond jauni, vignettage, pliure, bruit,
poussières, flou, légère rotation. On peut ensuite comparer `plan2cad vectorize scan-N.png --dpi 300` au résultat du PDF vectoriel.

    python tools/pdf_to_fake_scans.py planche.pdf dossier_sortie [--dpi 300]
Les scans sont numérotés dans l'ordre de lecture (lignes de haut en bas, puis de gauche à droite) ; meta.json garde leur emprise (pt).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan2cad import vectorpdf as vp            # noqa: E402
from plan2cad.demo import _scan_effects         # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plans = vp.split_plans(vp.load(a.pdf))
    subprocess.run(["pdftoppm", "-r", str(a.dpi), "-gray", "-png", "-singlefile", a.pdf, str(out / "page")], check=True)
    page = cv2.imread(str(out / "page.png"), 0)
    k = a.dpi / 72.0
    rng = np.random.default_rng(a.seed)
    # lecture : on regroupe les plans par rangée (tolérance = la moitié de la hauteur d'un plan)
    h = float(np.median([bb[3] - bb[1] for bb, _ in plans]))
    plans = sorted(plans, key=lambda p: (int(p[0][1] // (0.5 * h)), p[0][0]))
    meta = []
    for i, (bb, _idx) in enumerate(plans, 1):
        m = 40
        x0, y0, x1, y1 = [int(round(v * k)) for v in (bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m)]
        crop = page[max(y0, 0):y1, max(x0, 0):x1]
        scan, _ = _scan_effects(crop, rng, angle=0.35 * (1 if i % 2 else -1), scale=1.0, tx=6, ty=-4)
        cv2.imwrite(str(out / f"scan-{i}.png"), scan)
        meta.append(dict(i=i, bbox_pt=[float(v) for v in bb], size_px=list(scan.shape[::-1]), dpi=a.dpi))
        print(f"scan-{i}.png  {scan.shape[1]}x{scan.shape[0]}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
