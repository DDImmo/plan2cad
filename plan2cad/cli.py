"""Ligne de commande : `plan2cad ...`"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .layers import CHARTER, DISCIPLINES
from .pipeline import Options, run_manifest

DISCS = ("arc", "elec", "plumb", "cvc", "gaz")
LOTS = DISCS + ("color",)


def _scale(s: str):
    """100 | 1:100 | auto | 1/4"=1' (pouce par pied, plans américains : 1/4" = 1' donne 1:48)."""
    import re
    t = s.strip().lower().replace("in", '"').replace("ft", "'")
    if t == "auto":
        return "auto"
    m = re.fullmatch(r"(?:1\s*:\s*)?(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)\s*\"\s*=\s*1\s*'", t)
    if m:
        return 12.0 / (int(m.group(1)) / int(m.group(2)))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*\"\s*=\s*1\s*'", t)
    if m:
        return 12.0 / float(m.group(1))
    raise argparse.ArgumentTypeError("échelle : 100, 1:100, auto, ou 1/4\"=1' (plans en pieds)")


def _length(s: str) -> float:
    """Longueur connue : 17.4 (m), 17.4m, 1740cm, 57ft, 57'6\"."""
    import re
    t = s.strip().lower().replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)?", t)
    if m:
        return float(m.group(1)) * {"mm": 1.0, "cm": 10.0, "m": 1000.0, None: 1000.0}[m.group(2)]
    m = re.fullmatch(r"(\d+)\s*(?:ft|')\s*(?:(\d+)\s*(?:in|\")?)?", t)
    if m:
        return int(m.group(1)) * 304.8 + int(m.group(2) or 0) * 25.4
    raise argparse.ArgumentTypeError("longueur : 17.4 (m), 1740cm, 57ft, 57'6\"")


def _kv(s: str) -> dict:
    out = {}
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def _common(p: argparse.ArgumentParser):
    g = p.add_argument_group("vectorisation")
    g.add_argument("--dpi", type=float, default=400, help="résolution du scan (défaut 400)")
    g.add_argument("--scale", type=_scale, default="auto", help="dénominateur d'échelle du plan : 100 = 1:100. "
                   "Les DXF sont produits en mm réels. Défaut `auto` : échelle lue sur les cotes (≥ 3 concordantes), "
                   "sinon sur les surfaces de pièces ou la largeur des portes (PDF vectoriel), sinon 100.")
    g.add_argument("--known-length", type=_length, default=None, metavar="LONGUEUR",
                   help="longueur réelle connue du PLUS GRAND côté du bâtiment (murs, hors cotes), ex. 17.4 ou 57ft : "
                        "règle l'échelle quand aucune cote n'est lisible (écriture manuscrite…)")
    g.add_argument("--max-px", type=int, default=7000, help="côté max de travail en pixels (défaut 7000 ; "
                   "un A0 à 400 dpi est réduit automatiquement)")
    g.add_argument("--deskew", action="store_true", help="redresser le scan (±3°)")
    g.add_argument("--threshold", type=int, default=None, help="seuil noir/blanc 0-255 (défaut : automatique)")
    g.add_argument("--despeckle", type=float, default=0.4, metavar="MM", help="taille (mm papier) sous laquelle les taches sont supprimées")
    g.add_argument("--remove-text", nargs="*", metavar="LOT", default=None,
                   help="supprimer les petits éléments type texte ; sans argument = tous les lots, "
                        "ou une liste parmi arc elec plumb cvc")
    g.add_argument("--text-mm", type=float, default=4.0, help="hauteur max (mm papier) d'un 'texte'")
    g.add_argument("--close", type=int, default=0, metavar="PX", help="combler les trous de PX pixels dans les traits")
    g.add_argument("--simplify", type=float, default=0.35, metavar="MM", help="tolérance de simplification (mm papier)")
    g.add_argument("--min-length", type=float, default=3.0, metavar="MM", help="longueur min d'une polyligne (mm papier)")
    g.add_argument("--prune", type=float, default=1.5, metavar="MM", help="longueur max des épines supprimées (mm papier)")
    g.add_argument("--legend", metavar="legende.json", help="couleur -> calque, ex. {\"rouge\": \"ELEC-COURANT-FORT\", "
                   "\"jaune\": \"GAZ-RESEAU\"} (remplace/complète la légende par défaut)")
    g.add_argument("--wall-mode", choices=("auto", "solid", "double"), default="auto",
                   help="murs en double trait : auto (détection), solid (jamais), double (forcé)")
    g.add_argument("--wall-gap", type=float, default=None, metavar="MM",
                   help="écart (mm papier) entre les deux traits d'un mur ; force le mode double")
    g.add_argument("--keep-backdrop", action="store_true",
                   help="couleur : conserver le fond noir des plans de lots sur le calque Z-FOND-ARCHI")
    g.add_argument("--no-ocr", action="store_true", help="ne pas lire les textes (titres, noms de pièces)")
    g.add_argument("--texts", metavar="FICHIER.json", help="relecture des lettrages non lus (voir la planche *_a_relire.png)")
    g.add_argument("--no-regularize", action="store_true",
                   help="scans : ne pas redresser l'orientation ni rendre les murs exacts (droits, alignés)")
    g.add_argument("--raster", action="store_true",
                   help="traiter les PDF comme des images (par défaut, un PDF vectoriel est lu directement)")
    g.add_argument("--no-openings", action="store_true", help="ne pas chercher portes, fenêtres et escaliers")
    a = p.add_argument_group("calage")
    a.add_argument("--align", choices=("auto", "none"), default="auto", help="calage des lots sur le plan d'architecture")
    a.add_argument("--strip-backdrop", nargs="?", type=float, const=60.0, default=0.0, metavar="MM",
                   help="retirer des lots (élec, plomberie...) le fond d'architecture qui recouvre le plan maître, "
                        "tolérance MM en mm réels (60 par défaut si la valeur est omise)")
    t = p.add_argument_group("3D (optionnelle)")
    t.add_argument("--3d", dest="three_d", action="store_true", help="générer le modèle 3D (OBJ, STL, DXF 3DFACE)")
    t.add_argument("--wall-height", type=float, default=2700, help="hauteur des murs en mm (défaut 2700)")
    t.add_argument("--wall-thickness", type=float, default=None, help="épaisseur forcée des murs en mm (défaut : mesurée)")
    t.add_argument("--elev", type=_kv, default={}, metavar="elec=2400,plumb=300,cvc=2500",
                   help="altitude (mm) des réseaux")
    w = p.add_argument_group("interface web")
    w.add_argument("--serve", action="store_true", help="ouvrir l'interface web pendant le traitement")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true", help="ne pas ouvrir le navigateur automatiquement")


def _options(a) -> Options:
    rt = []
    if a.remove_text is not None:
        rt = ["*"] if len(a.remove_text) == 0 else a.remove_text
    legend = json.loads(Path(a.legend).read_text(encoding="utf-8")) if getattr(a, "legend", None) else {}
    return Options(texts_file=a.texts, regularize=not a.no_regularize, vector=not a.raster, known_length_mm=a.known_length, ocr=not a.no_ocr, openings=not a.no_openings, legend=legend, wall_mode=a.wall_mode, wall_gap_mm=a.wall_gap, keep_backdrop=a.keep_backdrop, dpi=a.dpi, scale=(100.0 if a.scale == 'auto' else a.scale), scale_auto=(a.scale == 'auto'), max_px=a.max_px, deskew=a.deskew, threshold=a.threshold,
                   despeckle_mm=a.despeckle, remove_text=rt, text_mm=a.text_mm, close_px=a.close,
                   strip_backdrop=a.strip_backdrop, simplify_mm=a.simplify, min_length_mm=a.min_length, prune_mm=a.prune, align=a.align,
                   three_d=a.three_d, wall_height=a.wall_height, wall_thickness=a.wall_thickness,
                   elevations=a.elev)


def _execute(project, opts, outdir, a, base_dir=None):
    """Lance le pipeline, avec le serveur web en parallèle si demandé."""
    from . import web
    srv = None
    if a.serve:
        srv = web.serve(outdir, a.port, open_browser=not a.no_browser, block=False)
        print(f"Interface web : http://127.0.0.1:{a.port}/")
    t0 = time.time()
    try:
        run_manifest(project, opts, outdir, base_dir=base_dir)
    except Exception as e:
        print(f"\nÉchec : {e}", file=sys.stderr)
        if srv:
            print("L'erreur est visible dans l'interface web. Ctrl+C pour quitter.")
            _wait()
        return 1
    print(f"\nTerminé en {time.time() - t0:.1f}s -> {Path(outdir).resolve()}")
    for f in sorted(Path(outdir).rglob("*")):
        if f.suffix.lower() in (".dxf", ".obj", ".stl") and "steps" not in f.parts:
            print(f"  {f.relative_to(outdir)}")
    if srv:
        print("L'interface web reste active. Ctrl+C pour quitter.")
        _wait()
    return 0


def _wait():
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print()


def cmd_run(a):
    base = None
    if a.project:
        pj = Path(a.project)
        project = json.loads(pj.read_text(encoding="utf-8"))
        project.setdefault("name", pj.stem)
        base = pj.parent
        levels = project["levels"] if "levels" in project else [lv for b in project["buildings"] for lv in b["levels"]]
    else:
        files = {d: getattr(a, d) for d in LOTS if getattr(a, d, None)}
        if not files:
            print("Indiquez au moins un plan : --arc / --elec / --plumb / --cvc / --gaz / --mixed, ou --project.",
                  file=sys.stderr)
            return 2
        project = dict(name=a.name or Path(next(iter(files.values()))).stem,
                       levels=[dict(name="RDC", files={d: str(Path(f).resolve()) for d, f in files.items()})])
        levels = project["levels"]
    for lv in levels:
        paths = list((lv.get("files") or {}).values()) + [s["file"] for s in lv.get("sources", [])]
        for f in paths:
            if not (Path(f) if Path(f).is_absolute() or not base else base / f).exists():
                print(f"Fichier introuvable : {f}", file=sys.stderr)
                return 2
    outdir = a.output or f"{project['name']}_plan2cad"
    return _execute(project, _options(a), outdir, a, base)


def cmd_vectorize(a):
    """Traitement par lots : chaque fichier -> un DXF."""
    out = Path(a.output or ".")
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for f in a.files:
        p = Path(f)
        if not p.exists():
            print(f"Fichier introuvable : {f}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n== {p.name}")
        work = out / f"{p.stem}_plan2cad"
        project = dict(name=p.stem, levels=[dict(name="RDC", files={a.lot: str(p.resolve())})])
        a_local = argparse.Namespace(**vars(a)); a_local.serve = False
        r = _execute(project, _options(a), str(work), a_local)
        if r == 0:
            shutil.copy(work / f"{p.stem}.dxf", out / f"{p.stem}.dxf")
        rc |= r
    return rc


def cmd_extrude(a):
    """DXF 2D existant -> 3D."""
    a.three_d = True
    project = dict(name=Path(a.dxf).stem, levels=[dict(name="RDC", files={"arc": str(Path(a.dxf).resolve())})])
    return _execute(project, _options(a), a.output or f"{Path(a.dxf).stem}_3d", a)


def cmd_scan(a):
    from .auto import scan_inputs
    from . import web
    opts = _options(a)
    out = a.output or "inventaire_plan2cad"
    srv = web.serve(out, a.port, open_browser=not a.no_browser, block=False) if a.serve else None
    try:
        manifest, run = scan_inputs(a.paths, out, opts, a.gap, a.scan_px, a.name)
    except Exception as e:
        print(f"\nÉchec : {e}", file=sys.stderr)
        return 1
    _print_manifest(manifest)
    print(f"\nManifeste : {Path(out, 'manifest.json').resolve()}  (relisible / corrigeable, puis : plan2cad build)")
    if srv:
        print("Interface web active. Ctrl+C pour quitter.")
        _wait()
    return 0


def _print_manifest(m):
    for b in m["buildings"]:
        print(f"\n  {b['name']}")
        for L in b["levels"]:
            print(f"    {L['name']}")
            for s in L["sources"]:
                print(f"      {s['id']:<24} {s['lot']:<6} confiance {s['confidence']:.0%}")
    if m["unassigned"]:
        print("\n  NON AFFECTÉS (à placer dans le manifeste) :")
        for s in m["unassigned"]:
            print(f"      {s['id']:<24} {s['lot']:<6} {' ; '.join(s['why'])[:90]}")
    if m["ignores"]:
        print(f"\n  {len(m['ignores'])} élément(s) ignoré(s) (cartouches, doublons)")


def cmd_build(a):
    from .auto import build_from_manifest
    from . import web
    out = a.output or str(Path(a.manifest).parent)
    srv = web.serve(out, a.port, open_browser=not a.no_browser, block=False) if a.serve else None
    try:
        build_from_manifest(a.manifest, out, _options(a), a.min_confidence)
    except Exception as e:
        print(f"\nÉchec : {e}", file=sys.stderr)
        return 1
    print(f"\nTerminé -> {Path(out).resolve()}")
    if srv:
        print("Interface web active. Ctrl+C pour quitter.")
        _wait()
    return 0


def cmd_auto(a):
    from .auto import scan_inputs, build_from_manifest
    from . import web
    opts = _options(a)
    out = a.output or "auto_plan2cad"
    srv = web.serve(out, a.port, open_browser=not a.no_browser, block=False) if a.serve else None
    try:
        print("== 1/2 Inventaire ==")
        manifest, _ = scan_inputs(a.paths, out, opts, a.gap, a.scan_px, a.name, adopt=True)
        _print_manifest(manifest)
        print("\n== 2/2 Vectorisation, calage, export ==")
        build_from_manifest(Path(out, "manifest.json"), out, _options(a), a.min_confidence)
    except Exception as e:
        print(f"\nÉchec : {e}", file=sys.stderr)
        return 1
    print(f"\nTerminé -> {Path(out).resolve()}  (vérifier les confiances faibles dans l'onglet Inventaire)")
    if srv:
        print("Interface web active. Ctrl+C pour quitter.")
        _wait()
    return 0


def cmd_serve(a):
    from . import web
    web.serve(a.dir, a.port, open_browser=not a.no_browser)
    return 0


def cmd_demo(a):
    from .demo import make_demo
    d = make_demo(Path(a.output) / "scans")
    print(f"Scans de démonstration -> {Path(a.output, 'scans').resolve()}")
    project = dict(name="demo", levels=[dict(name="RDC", files={k: str(Path(v).resolve()) for k, v in d["paths"].items()})])
    if not a.remove_text:
        a.remove_text = ["arc"]
    return _execute(project, _options(a), str(Path(a.output) / "resultat"), a)


def cmd_metres(a):
    from . import metres as mt
    for f in a.dxf:
        rooms, info, paths = mt.run_on_dxf(f, a.output, px_mm=a.px)
        if not rooms:
            print(f"{f} : {info.get('erreur', 'aucune pièce fermée trouvée')}")
            continue
        s = mt.summarize(rooms)
        print(f"{f} : {s['pieces']} pièces, {s['surface_totale_m2']} m², {s['portes']} portes, {s['fenetres']} fenêtres")
        if s.get("controlees"):
            print(f"  contrôle sur {s['controlees']} surfaces écrites : écart moyen {s['ecart_moyen_pct']} %, "
                  f"médian {s['ecart_median_pct']} %, max {s['ecart_max_pct']} % ({s['dans_3pct']} dans ±3 %)")
        for k, v in paths.items():
            print(f"  {k}: {v}")


def cmd_layers(a):
    print(f"{'Calque':<22}{'Couleur ACI':<13}Description")
    for k, (c, desc) in CHARTER.items():
        print(f"{k:<22}{c:<13}{desc}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="plan2cad", description="Scan de plan -> DXF multi-calques calés -> 3D (optionnel).")
    ap.add_argument("--version", action="version", version=f"plan2cad {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="chaîne complète : nettoyage, vectorisation, calques, calage, [3D]")
    for d in DISCS:
        r.add_argument(f"--{d}", metavar="FICHIER", help=f"plan {DISCIPLINES[d]['label']} (png, tif, jpg, pdf ou dxf)")
    r.add_argument("--color", dest="color", metavar="FICHIER",
                   help="plan en COULEUR mêlant plusieurs lots : séparés par la couleur selon la légende")
    r.add_argument("--project", metavar="projet.json", help="projet multi-niveaux / points d'ancrage (voir README)")
    r.add_argument("-o", "--output", help="dossier de sortie")
    r.add_argument("--name", help="nom du projet (préfixe des fichiers)")
    _common(r)
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("vectorize", help="vectoriser un ou plusieurs scans en DXF (traitement par lots)")
    v.add_argument("files", nargs="+")
    v.add_argument("--lot", choices=LOTS, default="arc", help="lot des fichiers (défaut arc ; color = lots séparés par couleur)")
    v.add_argument("-o", "--output", help="dossier de sortie")
    _common(v)
    v.set_defaults(fn=cmd_vectorize)

    e = sub.add_parser("extrude", help="extruder en 3D un DXF 2D existant")
    e.add_argument("dxf")
    e.add_argument("-o", "--output")
    _common(e)
    e.set_defaults(fn=cmd_extrude)

    inv = argparse.ArgumentParser(add_help=False)
    inv.add_argument("--gap", type=float, default=15.0, metavar="MM",
                     help="écart (mm papier) qui sépare deux plans d'une même planche (défaut 15)")
    inv.add_argument("--scan-px", type=int, default=3000, help="résolution de travail de l'inventaire")
    inv.add_argument("--name", help="nom du projet")
    inv.add_argument("--min-confidence", type=float, default=0.0, metavar="0-1",
                     help="build/auto : ignorer les sources dont la confiance est inférieure")

    sc = sub.add_parser("scan", parents=[inv], help="inventorier un dossier de plans (planches, bâtiments, niveaux, lots) -> manifest.json")
    sc.add_argument("paths", nargs="+", help="fichiers et/ou dossiers (récursif)")
    sc.add_argument("-o", "--output")
    _common(sc)
    sc.set_defaults(fn=cmd_scan)

    bd = sub.add_parser("build", parents=[inv], help="traiter un manifest.json (créé par scan, corrigé à la main si besoin)")
    bd.add_argument("manifest")
    bd.add_argument("-o", "--output")
    _common(bd)
    bd.set_defaults(fn=cmd_build)

    au = sub.add_parser("auto", parents=[inv], help="scan + build en une commande sur un dossier quelconque")
    au.add_argument("paths", nargs="+", help="fichiers et/ou dossiers (récursif)")
    au.add_argument("-o", "--output")
    _common(au)
    au.set_defaults(fn=cmd_auto)

    s = sub.add_parser("serve", help="ouvrir l'interface web sur un dossier de résultats")
    s.add_argument("dir", nargs="?", default=".")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("demo", help="générer des scans fictifs et lancer la chaîne dessus")
    d.add_argument("-o", "--output", default="plan2cad_demo")
    _common(d)
    d.set_defaults(fn=cmd_demo)

    m = sub.add_parser("metres", help="métrés d'un DXF plan2cad : murs -> pièces -> surfaces, périmètres, portes (JSON, CSV, DXF, PNG)")
    m.add_argument("dxf", nargs="+")
    m.add_argument("-o", "--output", help="dossier de sortie (défaut : à côté du DXF)")
    m.add_argument("--px", type=float, default=10.0, help="finesse de la grille en mm (défaut 10)")
    m.set_defaults(fn=cmd_metres)
    l = sub.add_parser("layers", help="afficher la charte de calques")
    l.set_defaults(fn=cmd_layers)
    return ap


def main(argv=None):
    for st in (sys.stdout, sys.stderr):      # console Windows (cp1252) : forcer l'UTF-8
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
