"""Métrés : murs -> pièces -> surfaces, périmètres, ouvertures.

Entrée : un DXF produit par plan2cad (mm réels). Les murs sont les faces du calque ARC-MURS-INT (+ fenêtres) ; les
portes et passages fermés par les segments d'ouverture du calque ARC-PORTE ; le reste (mobilier, cotes) est ignoré.

Méthode (raster, sans dépendance) :
  1. les faces de murs sont tracées sur une grille (10 mm/px par défaut) puis « fermées » (fermeture morphologique) : l'épaisseur
     d'un mur en double trait devient pleine, sans qu'il faille connaître son épaisseur ;
  2. les segments d'ouverture des portes ferment les passages ;
  3. les composantes connexes de l'espace libre sont les pièces ; celles qui touchent le bord de la grille (extérieur) sont écartées ;
  4. surface nette (face intérieure des murs), périmètre, contour, nombre de portes / fenêtres, libellé lu dans la pièce ;
  5. si un libellé « 5,27 m² » figure sur le plan, il est comparé à la surface calculée : c'est le contrôle de justesse.

Limites assumées : la fermeture arrondit légèrement les angles rentrants des pièces (≈ 7 cm de périmètre en moins par angle droit,
< 0,02 m² de surface) ; un espace ouvert sans mur ni porte (open space qui communique largement avec un couloir) forme UNE pièce avec lui ;
la surface est celle de l'espace libre, mobilier compris (comme un métré de surface utile).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

WALL_LAYERS = ("ARC-MURS-INT", "ARC-MURS-EXT")
WINDOW_LAYER = "ARC-FENETRE"
DOOR_LAYER = "ARC-PORTE"

_AREA_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s*m\s*[²2?^'’]", re.I)


def printed_area(s: str):
    """« 5,27 m² » -> 5.27 ; None si le texte n'est pas une surface."""
    m = _AREA_RE.search(s or "")
    return float(m.group(1).replace(",", ".")) if m else None


def _draw(img, pts, closed, px, org, thick=1):
    q = np.round((np.asarray(pts, float) - org) / px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [q], closed, 255, thick, cv2.LINE_8)


def compute_rooms(ents, texts=(), px_mm: float = 10.0, wall_gap_mm: float = 320.0, min_room_m2: float = 0.4,
                  wall_layers=WALL_LAYERS, door_layer=DOOR_LAYER, window_layer=WINDOW_LAYER):
    """ents : sortie de dxf.read_dxf ; texts : [dict(s, pos, height, angle)].
    Renvoie (pièces, infos). Chaque pièce : id, surface_m2, perimetre_m, contour (mm), portes, fenetres, libelle, surface_lue_m2, ecart_pct."""
    walls = [e for e in ents if e["type"] == "poly" and (e["layer"] in wall_layers or e["layer"] == window_layer)]
    doors = [e for e in ents if e["type"] == "poly" and e["layer"] == door_layer and len(e["pts"]) == 2]
    wins = [e for e in ents if e["type"] == "poly" and e["layer"] == window_layer]
    if not walls:
        return [], dict(erreur="aucun mur (calque %s) dans le DXF" % "/".join(wall_layers))
    allp = np.vstack([e["pts"] for e in walls])
    margin = 1500.0
    org = allp.min(0) - margin
    size = np.ceil((allp.max(0) + margin - org) / px_mm).astype(int)
    W, H = int(size[0]), int(size[1])

    wall = np.zeros((H, W), np.uint8)
    for e in walls:
        _draw(wall, e["pts"], e["closed"], px_mm, org)
    k = int(round(wall_gap_mm / px_mm)) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    wall_closed = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, kern)
    sealed = wall_closed.copy()
    for e in doors:
        _draw(sealed, e["pts"], False, px_mm, org, thick=3)
    free = (sealed == 0).astype(np.uint8)
    r = max(1, int(round(100.0 / px_mm)))                                     # coupe les filets < 20 cm (fins de mur ouverts)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    rooms = []
    areas_txt = [(printed_area(t["s"]), np.asarray(t["pos"], float)) for t in texts if printed_area(t["s"]) is not None]
    for i in range(1, n):
        x, y, w, h, npx = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:                      # touche le bord : extérieur
            continue
        mask = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        per_px = cv2.arcLength(c, True)
        # la ligne de mur (1 px) mord d'un demi-pixel dans la pièce : on le rend à la surface
        area_m2 = (npx + 0.5 * per_px) * px_mm * px_mm / 1e6
        if area_m2 < min_room_m2:
            continue
        ap = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float)
        poly_mm = ap * px_mm + org
        rooms.append(dict(label=i, area=area_m2, per=per_px * px_mm / 1000.0, poly=poly_mm, mask=mask, bbox=(x, y, w, h)))

    out = []
    dm = [(np.asarray(e["pts"]).mean(0), float(np.hypot(*(e["pts"][1] - e["pts"][0])))) for e in doors]
    wm = [np.asarray(e["pts"]).mean(0) for e in wins]
    reach = int(round(250.0 / px_mm))
    for rm in sorted(rooms, key=lambda r: (-r["poly"][:, 1].mean() // 1000, r["poly"][:, 0].mean())):
        x, y, w, h = rm["bbox"]
        pad = reach + 2
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
        near = cv2.dilate(rm["mask"][y0:y1, x0:x1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))

        def touches(p):
            px_ = int(round((p[0] - org[0]) / px_mm)) - x0
            py_ = int(round((p[1] - org[1]) / px_mm)) - y0
            return 0 <= px_ < near.shape[1] and 0 <= py_ < near.shape[0] and near[py_, px_] > 0

        def inside(p):
            px_ = int(round((p[0] - org[0]) / px_mm))
            py_ = int(round((p[1] - org[1]) / px_mm))
            return 0 <= px_ < W and 0 <= py_ < H and lab[py_, px_] == rm["label"]

        nd = sum(1 for c_, _w in dm if touches(c_))
        nw = sum(1 for c_ in wm if touches(c_))
        stated = [a for a, p in areas_txt if inside(p)]
        words = [t for t in texts if inside(np.asarray(t["pos"], float)) and printed_area(t["s"]) is None
                 and abs(float(t.get("angle", 0.0))) < 1e-6 and float(t.get("height", 0)) >= 150
                 and not re.fullmatch(r"[\d.,\s]+", t["s"].strip())]
        words.sort(key=lambda t: (-round(float(t["pos"][1]) / 400.0), float(t["pos"][0])))
        r_ = dict(id=len(out) + 1, surface_m2=round(rm["area"], 2), perimetre_m=round(rm["per"], 2), portes=nd, fenetres=nw,
                  libelle=" / ".join(t["s"].strip() for t in words)[:90], surface_lue_m2=None, ecart_pct=None,
                  centre_mm=[round(float(rm["poly"][:, 0].mean())), round(float(rm["poly"][:, 1].mean()))],
                  contour_mm=np.round(rm["poly"]).astype(int).tolist())
        if len(stated) == 1:
            r_["surface_lue_m2"] = stated[0]
            r_["ecart_pct"] = round(100.0 * (rm["area"] - stated[0]) / stated[0], 1)
        elif len(stated) > 1:
            r_["surface_lue_m2"] = stated
        out.append(r_)
    info = dict(px_mm=px_mm, grille=[W, H], pieces=len(out), portes_fermees=len(doors), fenetres=len(wins), origine_mm=org.tolist())
    return out, info


def summarize(rooms):
    ok = [r for r in rooms if isinstance(r["surface_lue_m2"], float)]
    s = dict(pieces=len(rooms), surface_totale_m2=round(sum(r["surface_m2"] for r in rooms), 1),
             portes=sum(r["portes"] for r in rooms), fenetres=sum(r["fenetres"] for r in rooms), controlees=len(ok))
    if ok:
        e = np.array([abs(r["ecart_pct"]) for r in ok])
        s.update(ecart_moyen_pct=round(float(e.mean()), 1), ecart_median_pct=round(float(np.median(e)), 1),
                 ecart_max_pct=round(float(e.max()), 1), dans_3pct=int((e <= 3).sum()), dans_10pct=int((e <= 10).sum()))
    big = [r for r in ok if r["surface_lue_m2"] >= 4.0 and r["surface_m2"] > 0]
    if len(big) >= 5:
        f = np.array([np.sqrt(r["surface_lue_m2"] / r["surface_m2"]) for r in big])
        med = float(np.median(f))
        spread = float(np.median(np.abs(f - med)) / med)
        s["facteur_echelle"] = round(med, 3)
        s["dispersion_pct"] = round(100 * spread, 1)
        # écart CONSTANT quelle que soit la taille de la pièce = erreur d'échelle (un biais d'épaisseur de mur,
        # lui, décroît avec la taille de la pièce)
        s["echelle_douteuse"] = bool(abs(med - 1) > 0.04 and spread < 0.05)
    return s


def write_outputs(rooms, info, outdir, stem, layer_colors=None):
    """JSON + CSV (+ DXF des contours et PNG de contrôle si possible). Renvoie les chemins."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    p = out / f"{stem}_metres.json"
    p.write_text(json.dumps(dict(resume=summarize(rooms), info=info, pieces=rooms), ensure_ascii=False, indent=1), encoding="utf-8")
    paths["json"] = str(p)
    p = out / f"{stem}_metres.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh, delimiter=";")
        wr.writerow(["id", "libelle", "surface_m2", "perimetre_m", "portes", "fenetres", "surface_lue_m2", "ecart_pct"])
        for r in rooms:
            wr.writerow([r["id"], r["libelle"], str(r["surface_m2"]).replace(".", ","), str(r["perimetre_m"]).replace(".", ","),
                         r["portes"], r["fenetres"], "" if r["surface_lue_m2"] is None else str(r["surface_lue_m2"]).replace(".", ","),
                         "" if r["ecart_pct"] is None else str(r["ecart_pct"]).replace(".", ",")])
    paths["csv"] = str(p)
    from .dxf import write_dxf
    ents = []
    for r in rooms:
        ents.append(("polyline", "ARC-PIECES", np.asarray(r["contour_mm"], float), True))
    p = out / f"{stem}_pieces.dxf"
    write_dxf(str(p), ents, dict(layer_colors or {}, **{"ARC-PIECES": 3}))
    paths["dxf"] = str(p)
    return paths


def render_png(rooms, ents, path, px_mm=40.0):
    """Image de contrôle : pièces colorées, surface calculée (et lue) écrite dedans."""
    allp = np.vstack([np.asarray(r["contour_mm"], float) for r in rooms])
    org = allp.min(0) - 800
    W, H = np.ceil((allp.max(0) + 800 - org) / px_mm).astype(int)
    im = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(3)

    def to_img(p):                                   # mm monde (y vers le haut) -> pixels image (y vers le bas)
        q = (np.asarray(p, float) - org) / px_mm
        q[..., 1] = H - 1 - q[..., 1]
        return q

    for r in rooms:
        q = np.round(to_img(r["contour_mm"])).astype(np.int32).reshape(-1, 1, 2)
        col = tuple(int(v) for v in rng.integers(150, 235, 3))
        cv2.fillPoly(im, [q], col)
    for e in ents:
        if e["type"] == "poly" and e["layer"] in WALL_LAYERS:
            q = np.round(to_img(e["pts"])).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(im, [q], e["closed"], (60, 60, 60), 1, cv2.LINE_AA)
    for r in rooms:
        cx, cy = to_img(r["centre_mm"])
        txt = f"{r['surface_m2']:.2f}"
        ok = r["surface_lue_m2"] is not None and isinstance(r["surface_lue_m2"], float)
        col = (0, 120, 0) if ok and abs(r["ecart_pct"]) <= 3 else ((0, 0, 200) if ok else (90, 90, 90))
        cv2.putText(im, txt, (int(cx) - 28, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if ok:
            cv2.putText(im, f"({r['surface_lue_m2']})", (int(cx) - 28, int(cy) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), im)
    return str(path)


def run_on_dxf(path, outdir=None, px_mm: float = 10.0, png: bool = True):
    """Métrés d'un DXF plan2cad. Écrit JSON / CSV / DXF des pièces (+ PNG de contrôle) à côté du DXF ou dans outdir."""
    from .dxf import read_dxf, read_dxf_texts
    ents, colors, _ = read_dxf(path)
    texts = read_dxf_texts(path)
    rooms, info = compute_rooms(ents, texts, px_mm=px_mm)
    if not rooms:
        return rooms, info, {}
    stem = Path(path).stem
    paths = write_outputs(rooms, info, outdir or Path(path).parent, stem, colors)
    if png:
        paths["png"] = render_png(rooms, ents, Path(outdir or Path(path).parent) / f"{stem}_metres.png")
    return rooms, info, paths
