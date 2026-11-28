"""Rendu d'un DXF plan2cad : image complète + un panneau par calque (+ planche).
Usage : python tools/render_layers.py plan_ARC.dxf sortie_dir [--width 1400]"""
import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan2cad.dxf import read_dxf
from plan2cad.layers import CHARTER

ACI = {1: (45, 50, 229), 2: (0, 162, 212), 3: (85, 157, 31), 4: (198, 165, 14), 5: (234, 91, 47), 6: (211, 38, 192),
       7: (140, 143, 138), 8: (144, 129, 123), 9: (196, 186, 180), 30: (0, 138, 240), 40: (0, 143, 201)}
ORDER = ["ARC-MOBILIER", "ARC-DETAIL", "ARC-COULEUR-ROUGE", "ARC-COULEUR-MAGENTA", "ARC-COULEUR-CYAN", "ARC-MURS-AXE",
         "ARC-MURS-INT", "ARC-MURS-EXT", "ARC-COTES", "ARC-PORTE", "ARC-FENETRE", "ARC-ESCALIER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf"); ap.add_argument("out"); ap.add_argument("--width", type=int, default=1400)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ents, _, _ = read_dxf(a.dxf)
    P = [e for e in ents if e["type"] == "poly"]
    by = collections.defaultdict(list)
    for e in P:
        by[e["layer"]].append(e)
    allp = np.vstack([e["pts"] for e in P]); lo, hi = allp.min(0), allp.max(0)
    col = lambda L: ACI.get(CHARTER.get(L, (7,))[0], (120, 120, 120))

    def draw(layers, W, thick=2):
        sc = W / (hi - lo).max(); h = int((hi - lo)[1] * sc) + 30; w = int((hi - lo)[0] * sc) + 30
        img = np.full((h, w, 3), 255, np.uint8)
        for L in layers:
            for e in by.get(L, []):
                q = (e["pts"] - lo) * sc + 15; q[:, 1] = h - q[:, 1]
                cv2.polylines(img, [q.astype(np.int32).reshape(-1, 1, 2)], e["closed"], col(L), thick, cv2.LINE_AA)
        return img

    layers = [L for L in ORDER if L in by] + [L for L in by if L not in ORDER and L in CHARTER]
    cv2.imwrite(str(out / "rendu_complet.png"), draw(layers, a.width))
    panels = []
    for L in layers:
        im = draw([L], 700)
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (245, 245, 245), -1)
        cv2.putText(im, f"{L}  ({len(by[L])})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), (200, 200, 200), 1)
        panels.append(im); cv2.imwrite(str(out / f"calque_{L}.png"), im)
    h = max(p.shape[0] for p in panels); w = max(p.shape[1] for p in panels)
    pad = lambda p: cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    blank = pad(np.full((10, 10, 3), 255, np.uint8))
    rows = [np.hstack([pad(p) for p in panels[i:i + 4]] + [blank] * (4 - len(panels[i:i + 4]))) for i in range(0, len(panels), 4)]
    cv2.imwrite(str(out / "calques_planche.png"), np.vstack(rows))
    print({L: len(by[L]) for L in layers})


if __name__ == "__main__":
    main()
