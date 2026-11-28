"""Inventaire automatique d'un dossier de plans : n'importe quelle organisation.

  fichiers (png/tif/jpg/pdf multi-pages) -> planches -> plans individuels (zones)
  -> caractérisation (couleurs, murs, pièces) -> bâtiment / niveau / lot
  -> manifeste (manifest.json) relisible et corrigeable.

Chaque décision porte une confiance (0-1) et une explication : rien n'est silencieux.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from . import preprocess as pp
from . import vectorize as vz
from . import ocr as ocrmod
from . import vectorpdf as vp
from .align import auto_align, footprint_iou
from .model import Poly, bbox, transform_polys

SUPPORTED = pp.RASTER_EXT | {".pdf"}
STOREY_MM = 2700.0


# ------------------------------------------------------------------ fichiers
def _natural(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def collect_files(paths) -> list[tuple[Path, Path | None]]:
    """Renvoie [(fichier, racine)] ; la racine sert à lire les noms de dossiers."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*"), key=_natural):
                rel = f.relative_to(p)
                if (f.is_file() and f.suffix.lower() in SUPPORTED
                        and not any(x.startswith(".") or x == "steps" for x in rel.parts)):
                    try:                                         # un lien symbolique ne doit pas faire sortir du dossier
                        if not f.resolve().is_relative_to(p.resolve()):
                            continue
                    except (OSError, ValueError):
                        continue
                    out.append((f, p))
        elif p.is_file():
            out.append((p, p.parent))
    return out


# ------------------------------------------------------------------ indices dans les noms
LOT_WORDS = {
    "arc": {"archi", "architecte", "architecture", "architectural", "go", "grosoeuvre", "gros", "fond", "fonds"},
    "elec": {"elec", "electricite", "electrique", "courant", "cfo", "cfa", "eclairage", "ecl"},
    "plumb": {"plomb", "plomberie", "sanitaire", "sanitaires", "eau", "eaux", "ef", "ecs", "eu", "ev", "evac",
              "evacuation", "evacuations", "ep"},
    "cvc": {"cvc", "chauffage", "ventilation", "vmc", "clim", "climatisation", "cta"},
    "gaz": {"gaz"},
}


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def parse_tokens(path: Path, root: Path | None) -> dict:
    """Bâtiment / niveau / lot lus dans le chemin (dossiers + nom). Rien d'inventé : seulement ce qui est écrit."""
    try:
        rel = path.relative_to(root) if root else Path(path.name)
    except ValueError:
        rel = Path(path.name)
    return _tokens_from_string(_strip(" / ".join(list(rel.parts[:-1]) + [rel.stem])))


def _tokens_from_string(text: str) -> dict:
    low = text.lower()
    words = set(re.split(r"[^a-z0-9+]+", low))
    tok: dict = {}
    hits = [lot for lot, ws in LOT_WORDS.items() if words & ws]
    if len(hits) == 1:
        tok["lot"] = hits[0]
    elif len(hits) > 1:
        tok["lot_ambigu"] = hits

    lv = None
    if re.search(r"(?<![a-z0-9])(rdc|rez[ _-]?de[ _-]?chaussee)(?![a-z0-9])", low):
        lv = ("RDC", 0)
    else:
        m = (re.search(r"(?<![a-z0-9])r\s*\+\s*(\d+)(?![0-9])", low)
             or re.search(r"(?<![a-z0-9])r(\d+)(?![a-z0-9])", low)
             or re.search(r"(?:etage|niveau|niv)[ _-]*\+?(\d+)(?![0-9])", low)
             or re.search(r"(?<![a-z0-9])n\s*\+\s*(\d+)(?![0-9])", low))
        if m:
            n_ = int(m.group(1))
            lv = ("RDC", 0) if (n_ == 0 and re.search(r"(?:etage|niveau|niv)[ _-]*\+?0", low)) else (f"R+{n_}", n_)
        else:
            m = re.search(r"(?<![a-z0-9])(?:ss|sous[ _-]?sol)[ _-]?(\d*)(?![a-z0-9])", low)
            if m:
                n = int(m.group(1) or 1)
                lv = (f"SS{n}", -n)
    if lv:
        tok["level"], tok["level_idx"] = lv

    m = re.search(r"(?i:(?<![A-Za-z])(bat(?:iment)?|bloc|tour|aile|corps))[ _\-]*([A-Za-z]|\d{1,2})(?![A-Za-z])", text)
    if m:
        kind = {"bat": "Bât", "batiment": "Bât", "bloc": "Bloc", "tour": "Tour", "aile": "Aile",
                "corps": "Corps"}[m.group(1).lower()]
        tok["building"] = f"{kind} {m.group(2).upper()}"
    return tok


# ------------------------------------------------------------------ données
@dataclass
class Region:
    id: str
    file: str
    page: int
    crop: list                      # fractions (x0, y0, x1, y1) de la page
    kind: str = "plan"              # plan | cartouche | annexe
    size_mm: list = field(default_factory=list)      # emprise réelle (mm) à l'échelle supposée
    features: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    lot: str | None = None
    building: str | None = None
    level: str | None = None
    level_idx: int | None = None
    conf: float = 0.0
    why: list = field(default_factory=list)
    thumb: str | None = None
    skip: bool = False
    # internes (non sérialisés)
    _geom: list = field(default_factory=list, repr=False)       # géométrie de calage (fond ou tout)
    _polys: list = field(default_factory=list, repr=False)
    _text_why: list = field(default_factory=list, repr=False)


def _public(r: Region) -> dict:
    d = asdict(r)
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ------------------------------------------------------------------ analyse d'une planche
def _quick_polys(ink, mm_px, opts):
    skel = vz.skeleton_of(ink)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    br, jc = vz.trace_skeleton(skel, dt)
    thick = np.array([b["width_px"] for b in br]) if br else np.array([0.0])
    prune = max(opts.prune_mm / mm_px, 0.6 * float(np.percentile(thick, 90)))
    pl = vz.build_polylines(br, jc, prune, max(1.0, opts.simplify_mm / mm_px), opts.min_length_mm / mm_px)
    k = mm_px * opts.scale
    H = ink.shape[0]
    return [Poly(np.stack([p[:, 0] * k, (H - p[:, 1]) * k], 1), "Q", c, w * k, "") for p, c, w in pl]


def _features(flat_c, ink_c, mm_px, opts, color):
    """Caractérise un plan : couleurs, épaisseur des murs, pièces fermées, double trait."""
    h, w = ink_c.shape
    inkpx = int((ink_c > 0).sum())
    f = dict(ink_pct=round(100 * inkpx / max(1, h * w), 2))
    dt = cv2.distanceTransform(ink_c, cv2.DIST_L2, 3)
    f["thick_frac"] = round(float(((dt * 2 * mm_px >= 0.8) & (ink_c > 0)).sum() / max(1, inkpx)), 3)
    k = max(3, int(8.0 / mm_px))                                 # ferme les portes (~8 mm papier)
    closed = cv2.morphologyEx(ink_c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    nb, lab, st, _ = cv2.connectedComponentsWithStats((closed == 0).astype(np.uint8), connectivity=4)
    areas, rooms = [], 0
    for i in range(1, nb):
        x, y, ww, hh, a = st[i]
        if x > 0 and y > 0 and x + ww < w and y + hh < h and a * mm_px * mm_px >= 100:
            rooms += 1
            areas.append(a)
    f["rooms"] = rooms
    f["room_cv"] = round(float(np.std(areas) / np.mean(areas)), 2) if len(areas) >= 3 else None
    gap, diag = vz.detect_wall_gap(ink_c, mm_px)
    f["double_trait"] = gap is not None
    ncomp, _, cst, _ = cv2.connectedComponentsWithStats(ink_c, connectivity=8)
    md = np.maximum(cst[1:, cv2.CC_STAT_WIDTH], cst[1:, cv2.CC_STAT_HEIGHT]) * mm_px
    f["text_ratio"] = round(float((md < 4.0).sum() / max(1, len(md))), 2)
    masks = None
    if color:
        masks = pp.classify_ink(flat_c, ink_c)
        tot = max(1, sum(int((m > 0).sum()) for m in masks.values()))
        f["colors"] = {n: round(float((m > 0).sum() / tot), 3) for n, m in masks.items()}
    return f, masks


def _arch_score(f: dict, tokens: dict) -> float:
    s = 0.0
    if f["thick_frac"] >= 0.35:
        s += 0.55
    elif f["thick_frac"] >= 0.2:
        s += 0.35                                 # murs hachurés / dessinés à la main : traits moins pleins
    if f["double_trait"]:
        s += 0.45
    if f["rooms"] >= 10:
        s += 0.35
    elif f["rooms"] >= 3:
        s += 0.25
    elif f["rooms"] == 2:
        s += 0.1
    if f.get("pieces_texte", 0) >= 2:
        s += 0.3                                  # noms de pièces écrits sur le plan
    colors = f.get("colors")
    if colors:
        nonblack = sum(v for n, v in colors.items() if n != "noir")
        s += -0.4 if nonblack >= 0.08 else 0.1
    lot = tokens.get("lot")
    if lot == "arc":
        s += 0.5
    elif lot:
        s -= 0.6
    return float(min(1.0, max(0.0, s)))


def _xycut(mask: np.ndarray, gap_px: int, min_px: int = 8):
    """Découpe XY récursive : coupe aux bandes blanches (lignes / colonnes sans encre) d'au moins `gap_px`.
    Renvoie des boîtes (x0, y0, x1, y1) dans le repère de `mask`."""
    def cuts(profile):
        on = np.nonzero(profile)[0]
        if len(on) == 0:
            return []
        segs, start, prev = [], on[0], on[0]
        for v in on[1:]:
            if v - prev - 1 >= gap_px:
                segs.append((start, prev + 1)); start = v
            prev = v
        segs.append((start, prev + 1))
        return segs

    def rec(x0, y0, x1, y1, depth, axis):
        sub = mask[y0:y1, x0:x1]
        if sub.size == 0 or not sub.any():
            return []
        for ax in (axis, 1 - axis):
            prof = sub.any(axis=1 - ax) if ax == 0 else sub.any(axis=0)         # ax 0 : lignes ; ax 1 : colonnes
            segs = cuts(prof)
            if len(segs) > 1 and depth < 8:
                out = []
                for a, b in segs:
                    if ax == 0:
                        out += rec(x0, y0 + a, x1, y0 + b, depth + 1, 1 - ax)
                    else:
                        out += rec(x0 + a, y0, x0 + b, y1, depth + 1, 1 - ax)
                return out
        ys, xs = np.nonzero(sub)
        return [(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)]

    return rec(0, 0, mask.shape[1], mask.shape[0], 0, 0)


def analyse_sheet(path: Path, page: int, root, opts, rid_prefix: str, gap_mm: float = 15.0,
                  scan_px: int = 3000, min_plan_mm: float = 60.0, thumbs_dir: Path | None = None):
    """Découpe une planche en plans et les caractérise. Renvoie (regions, sheet_info)."""
    img, info = pp.load_image(str(path), opts.dpi, scan_px, page, None)
    mm_px = info["mm_per_px"]
    color = info["is_color"]
    flat = pp.flatten(img)
    del img
    dark = pp.darkness(flat)
    # détection des zones : seuil sensible (le fond est normalisé à ~255), pour ne pas perdre les traits clairs
    # d'un plan de lot posé à côté de plans très noirs. La caractérisation refait un seuil propre à chaque plan.
    ink = (dark < 205).astype(np.uint8) * 255
    ink, _ = pp.clean_ink(ink, mm_px, opts.despeckle_mm)
    H, W = ink.shape

    # cadre de la planche : fin rectangle quasi pleine page QUI LAISSE du contenu une fois retiré.
    # (le mur extérieur d'un plan unique occupe aussi presque toute la page, mais il est épais et rien ne reste)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    frame = []
    total = int(st[1:, cv2.CC_STAT_AREA].sum()) if n > 1 else 0
    for i in range(1, n):
        if st[i, 2] >= 0.8 * W and st[i, 3] >= 0.8 * H:
            a = int(st[i, 4])
            width_mm = a / max(1, 2 * (st[i, 2] + st[i, 3])) * mm_px
            if width_mm <= 1.5 and total - a >= 0.05 * a:
                frame.append(i)
    if frame:
        ink = np.where(np.isin(lab, frame), 0, ink).astype(np.uint8)
    del lab

    g = max(3, int(gap_mm / mm_px))
    # seuls les éléments « de taille plan » (>= 30 mm) servent de germes : les textes, légendes et points parasites
    # ne doivent pas faire pont entre deux plans voisins ; ils sont rattachés ensuite à la zone qui les contient.
    nc, labc, stc, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = np.zeros(nc, bool)
    big[1:] = np.maximum(stc[1:, cv2.CC_STAT_WIDTH], stc[1:, cv2.CC_STAT_HEIGHT]) * mm_px >= 30.0
    seeds = (big[labc] * 255).astype(np.uint8)
    del labc
    dil = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (g, g)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    tokens = parse_tokens(path, root)

    phr = []
    if getattr(opts, "ocr", True) and ocrmod.available():
        phr, _k = ocrmod.read_texts(ink, dark, mm_px)
    pending: list[Region] = []
    regions: list[Region] = []
    annexes = 0
    order = sorted(range(1, n2), key=lambda i: (st2[i, 1] // max(1, int(H * 0.25)), st2[i, 0]))
    zones: list = []
    for i in order:
        x, y, w, h, _a = st2[i]
        blob = (lab2[y:y + h, x:x + w] == i) & (ink[y:y + h, x:x + w] > 0)
        if not blob.any():
            continue
        ys, xs = np.nonzero(blob)
        x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        bw, bh = x1 - x0, y1 - y0
        if max(bw, bh) * mm_px < min_plan_mm or bw * bh < 0.02 * W * H:
            annexes += 1
            continue
        if min(bw, bh) * mm_px >= 70.0:
            # des dessins voisins reliés par une cote ou un trait (façade au-dessus d'un plan, deux niveaux côte à côte)
            # sont séparés si une bande blanche d'au moins 7 mm les sépare
            boxes = _xycut(blob, max(3, int(7.0 / mm_px)))
            boxes = [(x + a, y + b, x + c, y + d) for a, b, c, d in boxes]
            big = [b_ for b_ in boxes if min(b_[2] - b_[0], b_[3] - b_[1]) * mm_px >= 25.0
                   and (b_[2] - b_[0]) * (b_[3] - b_[1]) >= 0.01 * W * H]
            if len(big) >= 2:
                for (bx0, by0, bx1, by1) in big:
                    zones.append((bx0, by0, bx1, by1, i))
                continue
        zones.append((x0, y0, x1, y1, i))

    for (x0, y0, x1, y1, i) in zones:
        bw, bh = x1 - x0, y1 - y0
        pad = int(2.0 / mm_px)
        cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        flat_c = flat[cy0:cy1, cx0:cx1]
        own, _ = pp.binarize(dark[cy0:cy1, cx0:cx1], opts.threshold)          # seuil propre à ce plan
        own = np.where(lab2[cy0:cy1, cx0:cx1] == i, own, 0).astype(np.uint8)
        bm = np.zeros_like(own); bm[max(0, y0 - cy0):y1 - cy0, max(0, x0 - cx0):x1 - cx0] = 1
        own = (own * bm).astype(np.uint8)
        ink_c, _ = pp.clean_ink(own, mm_px, opts.despeckle_mm)
        reg = Region(id=f"{rid_prefix}-{len(regions) + 1}", file=str(Path(path).resolve()), page=page,
                     crop=[round(cx0 / W, 4), round(cy0 / H, 4), round(cx1 / W, 4), round(cy1 / H, 4)],
                     tokens=dict(tokens),
                     size_mm=[round(bw * mm_px * opts.scale), round(bh * mm_px * opts.scale)])
        f, masks = _features(flat_c, ink_c, mm_px, opts, color)
        reg.features = f
        touches = x1 >= 0.93 * W or y1 >= 0.93 * H
        table = f["rooms"] >= 6 and f["room_cv"] is not None and f["room_cv"] < 0.6
        if touches and bw * bh < 0.08 * W * H and (table or f["text_ratio"] > 0.7):
            reg.kind = "cartouche"
            reg.skip = True
            reg.why.append("cartouche / tableau de la planche (ignoré)")
            regions.append(reg)
            continue

        # géométrie légère pour le regroupement (fond noir si plan couleur, sinon tout)
        use = ink_c
        if masks and "noir" in masks and len(masks) > 1:
            use = masks["noir"]
        if f["double_trait"]:
            gk = max(3, int(2.0 / mm_px))
            use = cv2.morphologyEx(use, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk)))
        reg._geom = _quick_polys(use, mm_px, opts)
        pending.append(reg)
        regions.append(reg)

    # textes lus sur la planche : rattachés au plan le plus proche (titre au-dessus, légendes, noms de pièces)
    near = 40.0 / mm_px
    mine: dict[str, list] = {r.id: [] for r in pending}
    for p in phr:
        cx, cy = p["x"] + p["w"] / 2, p["y"] + p["h"] / 2
        best, bd = None, near
        for r in pending:
            x0, y0, x1, y1 = r.crop[0] * W, r.crop[1] * H, r.crop[2] * W, r.crop[3] * H
            d = max(x0 - cx, 0, cx - x1) ** 2 + max(y0 - cy, 0, cy - y1) ** 2
            d = d ** 0.5
            if d <= bd:
                best, bd = r, d
        if best is not None:
            mine[best.id].append(p)
    for reg in pending:
        ps = mine[reg.id]
        votes, why = ocrmod.lot_votes(ps)
        reg.features["textes"] = [p["text"] for p in ps][:20]
        reg.features["lot_texte"] = {k: round(v, 1) for k, v in votes.items()}
        reg.features["pieces_texte"] = len(ocrmod.room_labels(ps))
        reg._text_why = why
        # niveau / bâtiment écrits dans les titres du plan (si le chemin n'en dit rien)
        titres = " / ".join(_strip(p["text"]) for p in ps
                            if re.search(r"(niveau|niv\b|etage|rdc|rez|r\s*\+|sous.?sol|batiment|bat\b|bloc|tour\b)", _strip(p["text"]).lower()))
        if titres:
            tt = _tokens_from_string(titres)
            for key in ("level", "level_idx", "building"):
                if key in tt and key not in reg.tokens:
                    reg.tokens[key] = tt[key]
            if "level" in tt:
                reg.why.append(f"titre lu sur le plan : « {titres} »")
        reg.features["arch_score"] = round(_arch_score(reg.features, reg.tokens), 2)
        _guess_lot(reg)
    sheet = dict(file=str(path), page=page, size_px=[W, H], frame=bool(frame), annexes=annexes,
                 color=color, mm_per_px=round(mm_px, 4))
    if thumbs_dir is not None:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        s = min(1.0, 1400 / max(W, H))
        prev = cv2.resize(flat if flat.ndim == 3 else cv2.cvtColor(flat, cv2.COLOR_GRAY2RGB),
                          (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        fn = f"{rid_prefix}.png"
        cv2.imwrite(str(thumbs_dir / fn), prev[..., ::-1])
        sheet["thumb"] = f"inventaire/sheets/{fn}"
        # vignettes des plans
        for r in regions:
            x0, y0, x1, y1 = r.crop
            c = flat[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            if c.size:
                cs = min(1.0, 360 / max(c.shape[:2]))
                c = cv2.resize(c, (max(1, int(c.shape[1] * cs)), max(1, int(c.shape[0] * cs))), interpolation=cv2.INTER_AREA)
                if c.ndim == 3:
                    c = c[..., ::-1]
                cv2.imwrite(str(thumbs_dir / f"{r.id}.png"), c)
                r.thumb = f"inventaire/sheets/{r.id}.png"
    return regions, sheet


def analyse_vector_sheet(path: Path, page: int, root, opts, rid_prefix: str, thumbs_dir: Path | None = None):
    """Planche issue d'un PDF vectoriel : plans = amas de tracés, titres et noms de pièces = texte exact du PDF."""
    d = vp.load(str(path), page)
    W, H = d["w"], d["h"]
    tokens = parse_tokens(path, root)
    regions: list[Region] = []
    plans = vp.split_plans(d)
    used_text: set = set()
    for k, (bb, idx) in enumerate(plans, 1):
        title = vp.title_for(d, bb)
        x0, y0, x1, y1 = bb
        pad = 6.0
        crop = [max(0.0, (x0 - pad) / W), max(0.0, (y0 - pad) / H), min(1.0, (x1 + pad) / W), min(1.0, (y1 + pad) / H)]
        r = Region(id=f"{rid_prefix}-{k}", file=str(path), page=page, crop=[round(c, 4) for c in crop])
        r.tokens = dict(tokens)
        if title:
            tt = _tokens_from_string(_strip(title))
            for key in ("level", "level_idx", "building"):
                if key in tt:
                    r.tokens[key] = tt[key]
            var = vp.variant_of(title)
            if var and not r.tokens.get("building"):
                r.tokens["building"] = var
            r.why.append(f"titre lu dans le PDF : « {title} »")
        ps = []
        for t in d["texts"]:
            b = t["bb"]
            if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1:
                ps.append(dict(text=t["s"], x=b[0], y=b[1], w=b[2] - b[0], h=b[3] - b[1], th=t["size"],
                               angle=0 if round(t["angle"]) % 180 == 0 else 90, conf=100))
        votes, why = ocrmod.lot_votes(ps)
        rooms = len(ocrmod.room_labels(ps))
        r.features = dict(vector=True, traces=len(idx), textes=[p["text"] for p in ps][:20], pieces_texte=rooms,
                          lot_texte={kk: round(v, 1) for kk, v in votes.items()}, rooms=rooms, arch_score=0.9)
        r.size_mm = [round((x1 - x0) * vp.PT_MM * float(opts.scale)), round((y1 - y0) * vp.PT_MM * float(opts.scale))]
        r._text_why = why
        top = sorted(((kk, v) for kk, v in votes.items() if kk != "arc"), key=lambda kv: -kv[1])
        tlot = top[0][0] if top and top[0][1] >= 3.0 and (len(top) == 1 or top[0][1] >= 2 * top[1][1]) else None
        if tlot and rooms < 3 and (not tokens.get("lot") or tokens["lot"] == tlot):
            r.lot, r.conf = tlot, 0.85
            r.why.append(f"texte du PDF → {tlot} (" + " ; ".join(why[:3]) + ")")
        else:
            r.lot = tokens.get("lot") or "arc"
            r.conf = 0.9 if (title and rooms >= 3) else 0.75
            r.why.append(f"PDF vectoriel : {len(idx)} tracés, {rooms} noms de pièces → "
                         + ("architecture" if r.lot == "arc" else r.lot))
        if not r.tokens.get("level"):
            r.conf = min(r.conf, 0.7)
        regions.append(r)
    sheet = dict(file=str(path), page=page, size_px=[int(W), int(H)], frame=False, annexes=0,
                 color=True, mm_per_px=round(vp.PT_MM, 4), vector=True)
    if thumbs_dir is not None:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        try:
            img, info = pp.load_image(str(path), 120, 1400, page, None)
            Hh, Ww = img.shape[:2]
            im3 = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            fn = f"{rid_prefix}.png"
            cv2.imwrite(str(thumbs_dir / fn), im3[..., ::-1])
            sheet["thumb"] = f"inventaire/sheets/{fn}"
            for r in regions:
                c0, c1, c2, c3 = r.crop
                c = im3[int(c1 * Hh):int(c3 * Hh), int(c0 * Ww):int(c2 * Ww)]
                if c.size:
                    cs = min(1.0, 360 / max(c.shape[:2]))
                    c = cv2.resize(c, (max(1, int(c.shape[1] * cs)), max(1, int(c.shape[0] * cs))), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(thumbs_dir / f"{r.id}.png"), c[..., ::-1])
                    r.thumb = f"inventaire/sheets/{r.id}.png"
        except Exception:
            pass
    return regions, sheet


def _guess_lot(r: Region):
    """Lot d'après le contenu et le nom. Confiance et raisons explicites."""
    f, tok = r.features, r.tokens
    colors = {n: v for n, v in (f.get("colors") or {}).items() if n != "noir" and v >= 0.03}
    arch = f["arch_score"]
    if colors:
        names = ", ".join(sorted(colors))
        if tok.get("lot") and tok["lot"] != "arc" and len(colors) == 1:
            r.lot, r.conf = tok["lot"], 0.85
            r.why.append(f"nom → {tok['lot']} ; une seule couleur ({names})")
        else:
            r.lot, r.conf = "color", 0.8
            r.why.append(f"plan en couleur ({names}) → lots séparés par la légende ; le noir sert de fond")
            if tok.get("lot") and len(colors) > 1:
                r.why.append(f"le nom indique « {tok['lot']} » mais {len(colors)} couleurs présentes")
        return
    votes = {k: v for k, v in (f.get("lot_texte") or {}).items() if k != "arc"}
    top = sorted(votes.items(), key=lambda kv: -kv[1])
    tlot = top[0][0] if top and top[0][1] >= 1.0 and (len(top) == 1 or top[0][1] >= 1.5 * top[1][1]) else None
    if tlot and (not tok.get("lot") or tok["lot"] == tlot):
        r.lot = tlot
        r.conf = 0.92 if tok.get("lot") == tlot else 0.85
        r.why.append(f"texte lu sur le plan → {tlot} (" + " ; ".join(getattr(r, "_text_why", [])[:3]) + ")")
        return
    if tlot and tok.get("lot") and tok["lot"] != tlot:
        r.why.append(f"le texte du plan évoque « {tlot} » mais le nom indique « {tok['lot']} » : nom retenu, à vérifier")
    if tok.get("lot"):
        r.lot = tok["lot"]
        r.conf = 0.9 if (tok["lot"] == "arc") == (arch >= 0.5) or arch in (0.0,) else 0.7
        r.why.append(f"nom du fichier/dossier → {tok['lot']}")
        if (tok["lot"] == "arc") != (arch >= 0.5):
            r.why.append("le contenu ne ressemble pas à ce lot : à vérifier")
            r.conf = 0.55
        return
    if (f["rooms"] <= 1 and f.get("ink_pct", 0) >= 10 and arch < 0.5 and not (f.get("lot_texte") or {})
            and not tok.get("lot")):
        r.kind = "illustration"
        r.skip = True
        r.lot = None
        r.why.append("croquis / perspective / façade (beaucoup d'encre, aucune pièce fermée) : ce n'est pas un plan, ignoré")
        return
    if arch >= 0.5:
        r.lot = "arc"
        r.conf = 0.55 + 0.4 * (arch - 0.5) * 2 / 2
        bits = []
        if f["thick_frac"] >= 0.35:
            bits.append("murs pleins")
        if f["double_trait"]:
            bits.append("murs en double trait")
        if f["rooms"] >= 3:
            bits.append(f"{f['rooms']} pièces fermées")
        r.why.append("architecture d'après le dessin : " + ", ".join(bits))
    else:
        r.lot = "autre"
        r.conf = 0.3
        r.why.append("plan de réseau en noir & blanc, lot non identifiable sans nom ni couleur : à préciser")


# ------------------------------------------------------------------ regroupement
def _compatible(a: Region, b: Region, tol=0.3) -> bool:
    (wa, ha), (wb, hb) = a.size_mm, b.size_mm
    ok = lambda x, y: abs(x - y) / max(x, y, 1) <= tol
    return (ok(wa, wb) and ok(ha, hb)) or (ok(wa, hb) and ok(ha, wb))


def _match(base: Region, other: Region, cache: dict, log=None):
    key = (base.id, other.id)
    if key in cache:
        return cache[key]
    res = None
    if base._geom and other._geom:
        try:
            sim, met = auto_align(base._geom, other._geom)
            moved = transform_polys(other._geom, sim)
            res = dict(fwd=met["inlier_ratio"], bwd=met["coverage"], iou=footprint_iou(base._geom, moved),
                       res_mm=met["median_residual_mm"])
        except Exception as e:                                      # géométrie dégénérée
            if log:
                log(f"  (calage impossible {base.id}/{other.id}: {e})")
    cache[key] = res
    return res


def group(regions: list[Region], log=print, opts=None):
    """Affecte bâtiment / niveau à chaque plan. Renvoie (buildings, unassigned)."""
    plans = [r for r in regions if r.kind == "plan" and not r.skip]
    bases = [r for r in plans if r.lot == "arc"]
    others = [r for r in plans if r.lot != "arc"]
    cache: dict = {}

    # --- 1. bâtiments : on regroupe les plans d'architecture qui ont la même emprise
    parent = {r.id: r.id for r in bases}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    notes: dict[str, list] = {r.id: [] for r in bases}
    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            ba, bb = a.tokens.get("building"), b.tokens.get("building")
            if ba and bb:
                if ba == bb:
                    parent[find(a.id)] = find(b.id)
                continue                                   # noms explicites : on ne regarde pas la géométrie
            if not _compatible(a, b, 0.35):
                continue
            m = _match(a, b, cache, log)
            # des emprises rectangulaires se ressemblent toutes : il faut une emprise quasi identique
            # (même bâtiment, autre niveau) ou, à défaut, un fort recouvrement des tracés eux-mêmes.
            lo = min(m["fwd"], m["bwd"]) if m else 0
            if m and ((m["iou"] >= 0.95 and lo >= 0.3) or (m["iou"] >= 0.8 and lo >= 0.65)):
                ra, rb = find(a.id), find(b.id)
                if ra == rb:
                    continue
                names = {x.tokens["building"] for x in bases if find(x.id) in (ra, rb) and x.tokens.get("building")}
                if len(names) > 1:                         # deux bâtiments nommés différemment : jamais fusionnés
                    continue
                parent[ra] = rb
                notes[b.id].append(f"même emprise que {a.id} (IoU {m['iou']:.2f})")
    comp: dict[str, list[Region]] = {}
    for r in bases:
        comp.setdefault(find(r.id), []).append(r)
    buildings: list[dict] = []
    used_names = set()
    for k, members in enumerate(comp.values(), 1):
        name = next((m.tokens["building"] for m in members if m.tokens.get("building")), None)
        if not name:
            name = f"Bâtiment {k}"
        while name in used_names:
            name += "'"
        used_names.add(name)
        # niveaux : un plan d'architecture par niveau
        levels: list[dict] = []
        for m in members:
            lv, idx = m.tokens.get("level"), m.tokens.get("level_idx")
            dup = None
            for L in levels:
                if lv and L["name"] == lv:
                    dup = L
                    break
                if not lv and L.get("anon"):
                    mm = _match(L["master"], m, cache, log)
                    if mm and mm["iou"] >= 0.9 and min(mm["fwd"], mm["bwd"]) >= 0.8:
                        dup = L
                        break
            if dup:
                m.skip = True
                m.why.append(f"doublon probable de {dup['master'].id} (même niveau) : ignoré")
                continue
            levels.append(dict(name=lv, idx=idx, master=m, sources=[m], anon=not lv))
        # niveaux sans nom : N1, N2... dans l'ordre d'apparition (confiance réduite)
        n_anon = 0
        for L in levels:
            if L["name"] is None:
                n_anon += 1
                L["name"] = "RDC" if len(levels) == 1 else f"N{n_anon}"
                L["idx"] = 0 if len(levels) == 1 else None
                L["master"].conf = min(L["master"].conf, 0.5 if len(levels) > 1 else 0.7)
                L["master"].why.append("niveau non indiqué" + (" : numérotation par ordre d'apparition" if len(levels) > 1 else " : un seul niveau, RDC supposé"))
        for m in members:
            m.building = name
        for L in levels:
            L["master"].level = L["name"]
            L["master"].level_idx = L["idx"]
            L["master"].why += notes.get(L["master"].id, [])
        buildings.append(dict(name=name, levels=levels, members=members))

    # --- 2. lots : rattachés par le nom, sinon par la géométrie (fond d'architecture commun)
    unassigned: list[Region] = []
    for r in others:
        cands = [(b, L) for b in buildings for L in b["levels"]]
        tb, tl = r.tokens.get("building"), r.tokens.get("level")
        if tb:
            c2 = [(b, L) for b, L in cands if b["name"] == tb]
            cands = c2 or cands
        if tl:
            c2 = [(b, L) for b, L in cands if L["name"] == tl]
            cands = c2 or cands
        scored = []
        for b, L in cands:
            if not _compatible(L["master"], r, 0.4):
                continue
            m = _match(L["master"], r, cache, log)
            if m:
                scored.append((m["fwd"] * 0.6 + m["iou"] * 0.4, m, b, L))
        scored.sort(key=lambda t: -t[0])
        if len(cands) == 1 and not scored:
            scored = [(0.2, None, cands[0][0], cands[0][1])]
        if not scored:
            r.why.append("aucun plan d'architecture compatible trouvé")
            unassigned.append(r)
            continue
        sc, m, b, L = scored[0]
        explicit = bool(tb and tl)
        if sc >= 0.5 or explicit:
            r.building, r.level, r.level_idx = b["name"], L["name"], L["idx"]
            r.conf = min(r.conf, 0.95 if explicit else (0.5 + 0.45 * min(1.0, sc)))
            if m:
                r.why.append(f"calé sur {L['master'].id} ({b['name']} / {L['name']}) : recouvrement {m['fwd']:.0%}, emprise {m['iou']:.0%}")
            if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08 and not explicit:
                r.conf = min(r.conf, 0.45)
                r.why.append(f"ambigu avec {scored[1][2]['name']} / {scored[1][3]['name']} (scores proches)")
            L["sources"].append(r)
        else:
            r.why.append(f"meilleure piste {b['name']} / {L['name']} mais recouvrement trop faible ({sc:.0%}) : non affecté")
            r.conf = min(r.conf, 0.25)
            unassigned.append(r)

    # niveaux : tri par indice (nom), sinon ordre d'apparition
    for b in buildings:
        b["levels"].sort(key=lambda L: (L["idx"] is None, L["idx"] if L["idx"] is not None else 0))
    return buildings, unassigned


# ------------------------------------------------------------------ manifeste
def _src(r: Region) -> dict:
    d = dict(id=r.id, file=r.file, lot=r.lot, confidence=round(r.conf, 2), why=r.why)
    if r.page > 1:
        d["page"] = r.page
    if not (r.crop[0] <= 0.001 and r.crop[1] <= 0.001 and r.crop[2] >= 0.999 and r.crop[3] >= 0.999):
        d["crop"] = r.crop
    if r.thumb:
        d["thumb"] = r.thumb
    return d


def build_manifest(name: str, buildings: list[dict], unassigned: list[Region], skipped: list[Region],
                   adopt: bool = False) -> dict:
    out = dict(name=name, buildings=[], unassigned=[], ignores=[])
    for b in buildings:
        lvls = []
        for k, L in enumerate(b["levels"]):
            idx = L["idx"] if L["idx"] is not None else k
            lvls.append(dict(name=L["name"], z=idx * STOREY_MM, height=STOREY_MM,
                             sources=[_src(r) for r in L["sources"] if not r.skip]))
        out["buildings"].append(dict(name=b["name"], levels=lvls))
    out["unassigned"] = [_src(r) | dict(lot=r.lot) for r in unassigned]
    if adopt and out["unassigned"]:
        # Chaîne automatique : un plan sans rattachement n'est pas perdu, il devient son propre bâtiment
        # (niveau RDC) ; à corriger dans le manifeste si besoin.
        for i, u in enumerate(out["unassigned"], 1):
            u["why"] = list(u.get("why", [])) + ["non rattaché : traité seul (bâtiment à renommer dans le manifeste)"]
            out["buildings"].append(dict(name=f"BAT-NA{i}" if len(out["unassigned"]) > 1 or out["buildings"] else "BAT-1",
                                         levels=[dict(name="RDC", z=0, height=STOREY_MM, sources=[u])]))
        out["unassigned"] = []
    out["ignores"] = [dict(id=r.id, file=r.file, kind=r.kind, why=r.why) for r in skipped]
    return out


def inventory_view(sheets: list[dict], regions: list[Region], manifest: dict) -> dict:
    """Données pour l'onglet « Inventaire » de l'interface web."""
    byid = {r.id: r for r in regions}
    out = []
    for s in sheets:
        regs = [_public(r) for r in regions if r.id.startswith(s["rid"] + "-")]
        out.append(dict(s, regions=regs))
    return dict(sheets=out, manifest=manifest)
