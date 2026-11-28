"""Chaîne « dossier de plans quelconque » : scan (inventaire) -> manifeste -> build."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
from PIL import Image

from . import preprocess as pp
from .inventory import (Region, analyse_sheet, analyse_vector_sheet, build_manifest, collect_files, group, inventory_view)
from .pipeline import Options, Run, run_manifest


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9+]+", "_", s).strip("_") or "x"


def _pages(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        return pp.pdf_pages(str(path))
    try:
        with Image.open(path) as im:
            return max(1, getattr(im, "n_frames", 1))
    except Exception:
        return 1


def _annotate(run_dir: Path, sheet: dict, regions: list[Region]):
    fn = run_dir / sheet["thumb"]
    img = cv2.imread(str(fn))
    if img is None:
        return
    H, W = img.shape[:2]
    for r in regions:
        x0, y0, x1, y1 = r.crop
        p0, p1 = (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H))
        if r.skip or r.kind != "plan":
            col = (150, 150, 150)
        elif not r.building:
            col = (60, 60, 220)
        else:
            col = (60, 170, 60) if r.conf >= 0.75 else (0, 150, 240) if r.conf >= 0.5 else (60, 60, 220)
        cv2.rectangle(img, p0, p1, col, 3)
        label = f"{r.id.rsplit('-', 1)[-1]} {r.lot or r.kind}"
        if r.building:
            label += f" | {r.building} {r.level or ''}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (p0[0], p0[1] - th - 10), (p0[0] + tw + 8, p0[1]), col, -1)
        cv2.putText(img, label, (p0[0] + 4, p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(fn), img)


def scan_inputs(paths, outdir, opts: Options, gap_mm: float = 15.0, scan_px: int = 3000,
                name: str | None = None, echo: bool = True, adopt: bool = False):
    """Inventorie un ou plusieurs fichiers / dossiers. Écrit outdir/00_inventaire/ (run web) et outdir/manifest.json."""
    out = Path(outdir)
    run_dir = out / "00_inventaire"
    run = Run(run_dir, "inventaire", opts, echo)
    files = collect_files(paths)
    if not files:
        run.finish(error="Aucun fichier de plan trouvé (png, tif, jpg, pdf).")
        raise RuntimeError("Aucun fichier de plan trouvé (png, tif, jpg, pdf).")
    regions: list[Region] = []
    sheets: list[dict] = []
    used: set[str] = set()
    try:
        for f, root in files:
            for page in range(1, _pages(f) + 1):
                rid = _slug(f.stem + (f"-p{page}" if page > 1 or _pages(f) > 1 else ""))
                while rid in used:
                    rid += "_"
                used.add(rid)
                with run.step(f.name, f"Analyse de la planche{f' (page {page})' if page > 1 else ''}") as st:
                    from . import vectorpdf as _vp
                    if getattr(opts, "vector", True) and _vp.is_vector(str(f), page):
                        regs, sheet = analyse_vector_sheet(f, page, root, opts, rid,
                                                           thumbs_dir=run_dir / "inventaire" / "sheets")
                    else:
                        regs, sheet = analyse_sheet(f, page, root, opts, rid, gap_mm, scan_px,
                                                    thumbs_dir=run_dir / "inventaire" / "sheets")
                    sheet["rid"] = rid
                    sheet["file"] = str(f)
                    sheets.append(sheet)
                    regions.extend(regs)
                    n_plan = sum(1 for r in regs if r.kind == "plan")
                    st.stats(plans=n_plan, cartouches=sum(1 for r in regs if r.kind == "cartouche"),
                             cadre=sheet["frame"], couleur=sheet["color"])
                    for r in regs:
                        if r.kind == "plan":
                            st.log(f"{r.id}: {r.lot} (confiance {r.conf:.0%}) – " + " ; ".join(r.why))
        with run.step("regroupement", "Bâtiments, niveaux et lots") as st:
            buildings, unassigned = group(regions, log=lambda m: None, opts=opts)
            skipped = [r for r in regions if r.skip]
            manifest = build_manifest(name or (out.name or "projet"), buildings, unassigned, skipped, adopt=adopt)
            st.stats(batiments=len(buildings),
                     niveaux=sum(len(b["levels"]) for b in buildings),
                     sources=sum(len(L["sources"]) for b in buildings for L in b["levels"]),
                     non_affectes=len(unassigned), ignores=len(skipped))
            for b in buildings:
                for L in b["levels"]:
                    st.log(f"{b['name']} / {L['name']} : " + ", ".join(f"{r.id}={r.lot}" for r in L["sources"] if not r.skip))
            for r in unassigned:
                st.log(f"NON AFFECTÉ {r.id}: " + " ; ".join(r.why))
        for sh in sheets:
            if sh.get("thumb"):
                _annotate(run_dir, sh, [r for r in regions if r.id.startswith(sh["rid"] + "-")])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        run.data["inventory"] = inventory_view(sheets, regions, manifest)
        run.finish()
    except Exception as e:
        run.finish(error=f"{type(e).__name__}: {e}")
        raise
    return manifest, run


def build_from_manifest(manifest_path, outdir, opts: Options, min_conf: float = 0.0, echo: bool = True,
                        allowed_roots=None):
    """Traite le manifeste (éventuellement corrigé à la main). Un dossier de résultats par bâtiment."""
    mp = Path(manifest_path)
    if mp.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError("Manifeste anormalement gros (> 20 Mo)")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    # un manifeste est une donnée NON FIABLE : ses chemins restent sous son dossier (ou les dossiers --allow-dir)
    roots = [mp.parent.resolve()] + [Path(r).resolve() for r in (allowed_roots or [])]
    if manifest.get("legend"):
        opts.legend = {**manifest["legend"], **(opts.legend or {})}
    skipped = 0
    for b in manifest["buildings"]:
        for L in b["levels"]:
            keep = []
            for s in L["sources"]:
                if s.get("skip") or s.get("confidence", 1.0) < min_conf:
                    skipped += 1
                    continue
                keep.append(s)
            L["sources"] = keep
        b["levels"] = [L for L in b["levels"] if L["sources"]]
    manifest["buildings"] = [b for b in manifest["buildings"] if b["levels"]]
    if not manifest["buildings"]:
        raise RuntimeError("Le manifeste ne contient aucun plan à traiter.")
    if skipped and echo:
        print(f"({skipped} source(s) ignorée(s) : skip ou confiance < {min_conf})")
    return run_manifest(manifest, opts, outdir, echo, base_dir=mp.parent, allowed_roots=roots)
