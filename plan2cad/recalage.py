"""Recalage de l'échelle sur les surfaces écrites (« 13 m² ») quand elles sont cohérentes entre elles.

Une erreur d'échelle multiplie toutes les longueurs par le même facteur, donc toutes les surfaces par son carré : le
rapport surface écrite / surface mesurée est alors CONSTANT quelle que soit la taille de la pièce (un biais d'épaisseur
de mur, lui, diminue avec la taille). Si ≥ 5 pièces de ≥ 4 m² donnent le même facteur (dispersion < 5 %) et qu'il
s'écarte de plus de 4 % de 1, les DXF du niveau sont remis à l'échelle (l'original est gardé en `.avant-recalage`).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

SCALED_CODES = {"10", "20", "30", "11", "21", "31", "40"}      # coordonnées, rayon, hauteur de texte


def rescale_dxf(src, dst, f: float):
    """Multiplie toutes les longueurs d'un DXF ASCII par f (paires code / valeur ; les angles ne changent pas)."""
    lines = Path(src).read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i in range(0, len(lines) - 1, 2):
        code, val = lines[i].strip(), lines[i + 1]
        if code in SCALED_CODES:
            try:
                val = f"{float(val) * f:.6f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
        out.append(lines[i])
        out.append(val)
    if len(lines) % 2:
        out.append(lines[-1])
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")


def check_dxf(path):
    """(facteur, dispersion %, nb pièces) ou None si les surfaces écrites ne permettent pas de conclure."""
    from . import metres as mt
    rooms, _, _ = mt.run_on_dxf(path, None, png=False, write=False)
    if not rooms:
        return None
    s = mt.summarize(rooms)
    if "facteur_echelle" not in s:
        return None
    return dict(facteur=s["facteur_echelle"], dispersion_pct=s["dispersion_pct"], pieces=s["controlees"],
                douteuse=s["echelle_douteuse"])


def recale_tree(outdir, log=print):
    """Parcourt `outdir`, recale chaque niveau dont le DXF d'architecture (*_ARC.dxf) est d'échelle douteuse.
    Les autres DXF du même niveau (même préfixe) suivent le même facteur. Renvoie la liste des recalages."""
    done = []
    for arc in sorted(Path(outdir).rglob("*_ARC.dxf")):
        if "steps" in arc.parts:
            continue
        try:
            res = check_dxf(arc)
        except Exception as e:                                  # un échec de contrôle ne doit jamais casser le résultat
            log(f"  recalage : contrôle impossible sur {arc.name} ({e})")
            continue
        if not res or not res["douteuse"]:
            if res:
                log(f"  recalage : {arc.name} cohérent avec les surfaces écrites (facteur {res['facteur']})")
            continue
        prefix = arc.name[:-len("_ARC.dxf")]
        touched = []
        for f in sorted(arc.parent.glob("*.dxf")):
            if f.name == f"{prefix}.dxf" or f.name.startswith(f"{prefix}_"):
                if f.name.endswith("_pieces.dxf"):
                    continue
                bak = f.with_name(f.name + ".avant-recalage")
                if not bak.exists():
                    shutil.copy2(f, bak)
                rescale_dxf(bak, f, res["facteur"])
                touched.append(f.name)
        rec = dict(niveau=prefix, **res, fichiers=touched)
        (arc.parent / f"{prefix}_recalage.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  recalage : {prefix} × {res['facteur']} ({res['pieces']} surfaces écrites concordantes, "
            f"dispersion {res['dispersion_pct']} %) -> {', '.join(touched)}")
        done.append(rec)
    return done
