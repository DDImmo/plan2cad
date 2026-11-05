"""Orchestration : enchaîne les étapes et journalise chacune (run.json) pour l'interface web."""
from __future__ import annotations

import json
import re
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from . import preprocess as pp
from . import vectorize as vz
from .align import auto_align, anchor_align, overlap_ratio, strip_backdrop
from .dxf import read_dxf, write_dxf
from .extrude import (ExtrudeOptions, build_mesh, merge_meshes, export_obj,
                      export_stl, export_dxf3d, mesh_to_json)
from .layers import (text_layer, ANCHOR_LAYER, CHARTER, DEFAULT_LEGEND, DISCIPLINES, aci_hex, disc_of_layer,
                     layer_color, split_walls_by_width)
from . import ocr as ocrmod
from . import openings as opn
from . import dimensions as dimmod
from .model import (Text, transform_texts, Poly, bbox, sim_apply, sim_compose, sim_params, transform_polys,
                    IDENTITY)


# ------------------------------------------------------------------ options
@dataclass
class Options:
    dpi: float = 400.0
    scale: float = 100.0              # dénominateur d'échelle (1:100 -> 100)
    max_px: int = 7000                # côté max de travail (A0 400 dpi = 18700 px, réduit)
    deskew: bool = False
    threshold: int | None = None
    despeckle_mm: float = 0.4         # poussières (mm papier)
    remove_text: list = field(default_factory=list)   # disciplines, ["*"] = toutes
    text_mm: float = 4.0
    close_px: int = 0
    simplify_mm: float = 0.35         # tolérance Douglas-Peucker (mm papier)
    min_length_mm: float = 3.0
    prune_mm: float = 1.5
    align: str = "auto"               # auto | none
    strip_backdrop: float = 0.0       # mm réels ; >0 : retire des lots le fond d'architecture (N&B)
    ocr: bool = True                  # lire les textes (Tesseract s'il est installé)
    openings: bool = True             # portes / fenêtres / escaliers
    known_length_mm: float | None = None   # plus grand côté réel du bâtiment (calibrage manuel)
    scale_auto: bool = False          # échelle lue sur les cotes (si ≥ 3 cotes concordent)
    keep_backdrop: bool = False       # couleur : conserver le fond noir des lots sur le calque Z-FOND-ARCHI
    legend: dict = field(default_factory=dict)   # couleur -> calque (surcharge DEFAULT_LEGEND)
    wall_mode: str = "auto"           # auto | solid | double  (murs en double trait)
    wall_gap_mm: float | None = None  # écart entre les deux traits (mm papier), force le mode double
    three_d: bool = False
    wall_height: float = 2700.0
    wall_thickness: float | None = None
    elevations: dict = field(default_factory=dict)


# ------------------------------------------------------------------ journal
class Run:
    def __init__(self, outdir, name, options: Options, echo=True):
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "steps").mkdir(exist_ok=True)
        self.echo = echo
        self.data = dict(name=name, status="running", created=time.time(), updated=time.time(),
                         options=asdict(options), steps=[], levels=[], outputs=[],
                         model3d=None, error=None)
        self._n = 0
        self.save()

    def save(self):
        self.data["updated"] = time.time()
        tmp = self.out / "run.json.tmp"
        tmp.write_text(json.dumps(self.data, ensure_ascii=False))
        os.replace(tmp, self.out / "run.json")

    @contextmanager
    def step(self, group: str, title: str):
        st = Step(self, group, title)
        self.data["steps"].append(st.d)
        self.save()
        t0 = time.time()
        if self.echo:
            print(f"  [{group}] {title} ...", end="", flush=True)
        try:
            yield st
            st.d["status"] = "done"
        except Exception:
            st.d["status"] = "error"
            st.d["logs"].append(traceback.format_exc())
            self.save()
            if self.echo:
                print(" ERREUR")
            raise
        finally:
            st.d["dur"] = round(time.time() - t0, 2)
            self.save()
        if self.echo:
            extra = " ".join(f"{k}={v}" for k, v in st.d["stats"].items() if not isinstance(v, (dict, list)))
            print(f" {st.d['dur']}s  {extra[:110]}")

    def add_output(self, label, path):
        self.data["outputs"].append(dict(label=label, file=str(Path(path).relative_to(self.out))))
        self.save()

    def finish(self, error=None):
        self.data["status"] = "error" if error else "done"
        self.data["error"] = error
        self.save()


class Step:
    def __init__(self, run: Run, group, title):
        self.run = run
        self.d = dict(id=len(run.data["steps"]), group=group, title=title, status="running",
                      t0=time.time(), dur=0, logs=[], stats={}, images=[])

    def log(self, msg):
        self.d["logs"].append(str(msg))
        self.run.save()

    def stats(self, **kw):
        self.d["stats"].update({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in kw.items()})

    def image(self, name, img, max_dim=1500):
        if img.ndim == 2 and img.dtype == bool:
            img = img.astype(np.uint8) * 255
        h, w = img.shape[:2]
        f = max_dim / max(h, w)
        if f < 1:
            interp = cv2.INTER_AREA
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=interp)
        self.run._n += 1
        fn = f"steps/{self.run._n:03d}_{self.d['group'].replace(':', '_')}_{name}.png"
        cv2.imwrite(str(self.run.out / fn), img)
        self.d["images"].append(dict(name=name, file=fn))
        self.run.save()


# ------------------------------------------------------------------ vectorisation d'une source
@dataclass
class SourceResult:
    id: str
    lot: str
    polys: list
    backdrop: list = field(default_factory=list)   # fond (noir) d'un plan de lot en couleur : sert au calage
    info: dict = field(default_factory=dict)
    texts: list = field(default_factory=list)


def _overlay(ink, polys_px, closed_flags, scale_dim=1500):
    h, w = ink.shape
    f = min(1.0, scale_dim / max(h, w))
    base = cv2.resize(ink, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    img = np.full((*base.shape, 3), 255, np.uint8)
    img[base > 60] = (215, 215, 215)
    rng = np.random.default_rng(3)
    for pts, closed in zip(polys_px, closed_flags):
        col = tuple(int(c) for c in rng.integers(30, 200, 3))
        cv2.polylines(img, [np.round(pts * f).astype(np.int32).reshape(-1, 1, 2)], closed, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(np.round(p * f).astype(int)), 2, (0, 0, 0), -1)
    return img


def _bgr(img):
    return img[..., ::-1] if img.ndim == 3 else img


def vectorize_mask(run: Run, tag: str, ink: np.ndarray, mm_px: float, opts: Options, label: str = ""):
    """masque d'encre -> polylignes en pixels [(pts, closed, width_px)]"""
    sfx = f" – {label}" if label else ""
    with run.step(tag, f"Squelette (lignes centrales){sfx}") as st:
        skel = vz.skeleton_of(ink)
        dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        st.stats(pixels_squelette=int(skel.sum()))
        ov = _overlay(ink, [], [])
        sk_small = cv2.resize(skel.astype(np.uint8) * 255, (ov.shape[1], ov.shape[0]), interpolation=cv2.INTER_AREA)
        ov[sk_small > 20] = (0, 0, 220)
        st.image("squelette" + (f"_{label}" if label else ""), ov)
    with run.step(tag, f"Tracé du graphe et simplification{sfx}") as st:
        branches, jcent = vz.trace_skeleton(skel, dt)
        thick = np.array([b["width_px"] for b in branches]) if branches else np.array([0.0])
        prune_px = max(opts.prune_mm / mm_px, 0.6 * float(np.percentile(thick, 90)))
        eps_px = max(1.0, opts.simplify_mm / mm_px)
        polys_px = vz.build_polylines(branches, jcent, prune_px, eps_px, opts.min_length_mm / mm_px)
        st.stats(branches=len(branches), jonctions=len(jcent), polylignes=len(polys_px),
                 sommets=int(sum(len(p[0]) for p in polys_px)), epine_px=round(prune_px, 1))
        st.image("vecteurs" + (f"_{label}" if label else ""),
                 _overlay(ink, [p[0] for p in polys_px], [p[1] for p in polys_px]))
    return polys_px


def _legend(opts: Options) -> dict:
    lg = dict(DEFAULT_LEGEND)
    lg.update(opts.legend or {})
    return lg


def vectorize_source(run: Run, tag: str, src: dict, opts: Options) -> SourceResult:
    """Une source = fichier (+ page, + zone) + lot. Lot : arc | elec | plumb | cvc | gaz | color | autre.
    `color` : plan de lots mélangés, séparés par la couleur selon la légende."""
    path, lot = src["file"], src.get("lot", "arc")
    sid = src.get("id", lot)
    if str(path).lower().endswith(".dxf"):
        return load_dxf_source(run, tag, src)

    with run.step(tag, "Chargement") as st:
        img, info = pp.load_image(path, opts.dpi, opts.max_px, src.get("page", 1), src.get("crop"))
        st.stats(fichier=Path(path).name, **({"page": src["page"]} if src.get("page", 1) > 1 else {}),
                 **({"zone": [round(c, 3) for c in src["crop"]]} if src.get("crop") else {}),
                 taille_px=f"{img.shape[1]}x{img.shape[0]}", dpi=round(info["dpi"], 1),
                 mm_papier_par_px=round(info["mm_per_px"], 4), couleur=info["is_color"])
        st.image("scan", _bgr(img))
    mm_px = info["mm_per_px"]
    color = info["is_color"]

    with run.step(tag, "Aplatissement du fond (ombres, pliures)") as st:
        flat = pp.flatten(img)
        del img
        st.image("fond_corrige", _bgr(flat))

    if opts.deskew:
        with run.step(tag, "Redressement") as st:
            ang = pp.estimate_skew(pp.darkness(flat))
            flat = pp.rotate_image(flat, ang)
            st.stats(angle_deg=round(ang, 3))
            st.image("redresse", _bgr(flat))

    with run.step(tag, "Binarisation (noir & blanc pur)") as st:
        ink, t = pp.binarize(pp.darkness(flat), opts.threshold)
        st.stats(seuil=t, encre_pct=round(100 * float((ink > 0).mean()), 2))
        st.image("binaire", 255 - ink)

    with run.step(tag, "Nettoyage (poussières / textes)") as st:
        rm = ("*" in opts.remove_text) or (lot in opts.remove_text)
        ink, s = pp.clean_ink(ink, mm_px, opts.despeckle_mm, rm, opts.text_mm, opts.close_px)
        st.stats(**s, textes_actif=rm)
        st.image("nettoye", 255 - ink)

    words, phr = [], []
    if opts.ocr:
        with run.step(tag, "Lecture des textes (OCR)") as st:
            if not ocrmod.available():
                st.stats(statut="Tesseract absent : textes non lus (apt install tesseract-ocr tesseract-ocr-fra)")
            else:
                gray = pp.darkness(flat)
                phr, kill = ocrmod.read_texts(ink, gray, mm_px)
                if kill is not None and kill.any():
                    ink = np.where(kill, 0, ink).astype(np.uint8)       # les lettres ne polluent plus le tracé
                words = phr
                votes, why = ocrmod.lot_votes(phr)
                rooms = ocrmod.room_labels(phr)
                st.stats(textes_lus=len(phr), verticaux=sum(1 for p_ in phr if p_["angle"] != 0),
                         lettres_retirees_du_dessin=int(kill.sum()) if kill is not None else 0,
                         pieces=", ".join(p["text"] for p in rooms[:12]) or "-",
                         lot_d_apres_texte=", ".join(f"{k} {v:.1f}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])) or "-")
                for wline in why[:6]:
                    st.log(wline)
                st.image("texte_retire", 255 - ink)
        info["ocr_phrases"] = [dict(text=p["text"], x=p["x"], y=p["y"], w=p["w"], h=p["h"], angle=p["angle"]) for p in phr]

    split = color and lot != "arc"
    masks = {"__ink__": ink}
    if split:
        with run.step(tag, "Séparation des couleurs") as st:
            masks = pp.classify_ink(flat, ink)
            st.stats(classes={k: f"{100 * (m > 0).sum() / max(1, (ink > 0).sum()):.0f} %" for k, m in masks.items()})
            st.image("couleurs", pp.class_preview(None, masks))
    del flat

    raw_ink = None                       # encre avant comblement des doubles traits (fenêtres = traits intérieurs au mur)
    if lot == "arc" and opts.wall_mode != "solid":
        with run.step(tag, "Murs en double trait ?") as st:
            if opts.wall_gap_mm:
                gap_mm, diag = opts.wall_gap_mm, {"force": True}
            else:
                gap_mm, diag = vz.detect_wall_gap(ink, mm_px)
                if opts.wall_mode == "double" and gap_mm is None:
                    gap_mm = 2.0
                    diag["repli"] = "écart 2 mm papier par défaut"
            st.stats(**diag)
            if gap_mm:
                raw_ink = ink.copy()
                k = int(np.ceil(gap_mm / mm_px)) + 2
                closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                # on ne garde du comblement que les LONGUES bandes entre deux traits de mur : les escaliers, symboles et
                # autres paires de traits rapprochés (marches, hachures) ne doivent pas être noyés dans le noir
                fill = cv2.bitwise_and(closed, cv2.bitwise_not(ink))
                nf, lf, sf, _ = cv2.connectedComponentsWithStats(fill, connectivity=8)
                lmin = 12.0 / mm_px
                keepf = np.zeros(nf, bool)
                keepf[1:] = np.maximum(sf[1:, cv2.CC_STAT_WIDTH], sf[1:, cv2.CC_STAT_HEIGHT]) >= lmin
                # une bande de mur a une épaisseur CONSTANTE (≈ écart entre les deux traits) ; un coin d'escalier ou un
                # éventail de marches se remplit en biseau (épaisseur variable) : rejeté
                dtf = cv2.distanceTransform(fill, cv2.DIST_L2, 3)
                ridge = (dtf >= cv2.dilate(dtf, np.ones((3, 3), np.uint8)) - 1e-3) & (dtf > 1.0)
                half = 0.5 * (gap_mm / mm_px)
                tot = np.bincount(lf[ridge], minlength=nf).astype(float)
                good = np.bincount(lf[ridge & (dtf >= 0.6 * half)], minlength=nf).astype(float)
                uniform = np.zeros(nf, bool)
                uniform[1:] = good[1:] >= 0.6 * np.maximum(tot[1:], 1)
                keepf &= uniform
                dropped = int((~keepf[1:]).sum())
                ink = cv2.bitwise_or(ink, (keepf[lf] * 255).astype(np.uint8))
                st.stats(petits_remplissages_ignores=dropped)
                # petits trous résiduels aux angles : ils créeraient des boucles parasites dans le squelette
                nb, lab, stt, _ = cv2.connectedComponentsWithStats(255 - ink, connectivity=4)
                small = stt[:, cv2.CC_STAT_AREA] < 1.5 * k * k            # trous d'angle seulement (pas les cases d'un escalier)
                small[0] = False
                small &= (stt[:, cv2.CC_STAT_WIDTH] < 1.6 * k) & (stt[:, cv2.CC_STAT_HEIGHT] < 1.6 * k)
                if small.any():
                    ink = np.where(small[lab], 255, ink).astype(np.uint8)
                    st.stats(trous_combles=int(small.sum()))
                masks = {"__ink__": ink}
                st.stats(decision=f"double trait : comblé sur {gap_mm:.2f} mm papier")
                st.image("murs_combles", 255 - ink)
            else:
                st.stats(decision="murs pleins (aucun traitement)")

    H = ink.shape[0]
    legend = _legend(opts)
    scale = float(opts.scale)
    per_class = {}
    for cname, m in masks.items():
        per_class[cname] = vectorize_mask(run, tag, m, mm_px, opts, "" if cname == "__ink__" else cname)

    wt = opn.wall_thickness(ink, mm_px) if lot == "arc" else None
    dims, cote_idx = [], {}
    if lot == "arc" and phr:
        with run.step(tag, "Cotes (échelle lue sur le plan)") as st:
            base = per_class.get("__ink__", [])
            dims = dimmod.find_dimensions(phr, base, mm_px, wt)
            est = dimmod.estimate_scale(dims, mm_px)
            if est:
                est["source"] = "cotes"
            else:
                est = dimmod.room_scale(phr, ink, mm_px, wt)
            st.stats(cotes_reliees=len(dims), cotes_lues=sum(1 for p_ in phr if dimmod.parse_value_sys(p_["text"])[0]),
                     dimensions_de_pieces=sum(1 for p_ in phr if dimmod.parse_room_label(p_["text"])))
            if est:
                st.stats(echelle_deduite=f"1:{est['scale']:g}", concordantes=f"{est['n']}/{est['of']}",
                         source=est.get("source", "cotes"), unites=est.get("systeme", "métrique"))
                if abs(est["scale"] / scale - 1) > 0.06:
                    if opts.scale_auto:
                        st.log(f"✔ échelle 1:{est['scale']:g} lue sur les cotes (au lieu de 1:{scale:g}) : appliquée")
                        scale = est["scale"]
                    else:
                        st.log(f"⚠ Les cotes indiquent 1:{est['scale']:g} alors que l'échelle utilisée est 1:{scale:g}. "
                               f"Relancer avec --scale {est['scale']:g} (ou --scale auto) : sinon dimensions et 3D faux.")
                else:
                    st.stats(verdict=f"cohérent avec 1:{scale:g}")
            else:
                st.stats(echelle_deduite="non déterminée (moins de 3 cotes concordantes)")
            for d in dims:
                for pj in d.get("polys", [d["poly"]]):
                    cote_idx[pj] = d["phrase"]
    if lot == "arc" and opts.known_length_mm and per_class.get("__ink__"):
        base_ = per_class["__ink__"]
        wall_ = [np.asarray(q_[0], float) for q_ in base_ if not wt or q_[2] >= 0.75 * wt] or [np.asarray(q_[0], float) for q_ in base_]
        allp_ = np.vstack(wall_)
        ext_ = float((allp_.max(0) - allp_.min(0)).max()) * mm_px
        if ext_ > 0:
            scale = opts.known_length_mm / ext_
            info["scale_from_known_length"] = round(scale, 2)
            std_ = min(dimmod.STD_SCALES, key=lambda x: abs(np.log(scale / x)))
            if abs(scale / std_ - 1) <= 0.03:
                scale = float(std_)                       # échelle normalisée la plus proche (à 3 % près)
    info["scale_used"] = scale
    k = mm_px * scale

    openings = []
    if lot == "arc" and opts.openings:
        with run.step(tag, "Portes et fenêtres") as st:
            openings, wt2 = opn.detect_openings(ink, mm_px, scale, wt, raw=raw_ink)
            if wt2 is None:
                st.stats(statut="pas de murs pleins reconnus : portes / fenêtres non cherchées")
            else:
                cnt = {}
                opn.tag_envelope(ink, openings, wt2, mm_px, scale)
                for o in openings:
                    cnt[o["kind"]] = cnt.get(o["kind"], 0) + 1
                st.stats(epaisseur_mur_mm=round(wt2 * mm_px * scale), **cnt)
                nb_env = sum(1 for o in openings if o.get("why", "").startswith("baie sur la façade"))
                if nb_env:
                    st.log(f"ℹ {nb_env} baie(s) sur la façade sans symbole classée(s) fenêtre : à vérifier (porte d'entrée possible)")
                ov = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
                for o in openings:
                    x0, y0, x1, y1 = o["rect"]
                    colr = {"porte": (200, 90, 20), "fenetre": (20, 160, 220), "ouverture": (130, 130, 130)}[o["kind"]]
                    cv2.rectangle(ov, (x0, y0), (x1, y1), colr, max(3, int(wt2 / 6)))
                f = min(1.0, 1500 / max(ov.shape[:2]))
                st.image("ouvertures", cv2.resize(ov, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    del masks, ink

    with run.step(tag, "Calques et mise à l'échelle") as st:
        polys, backdrop = [], []
        override: dict = {}
        stairs = []
        if lot == "arc" and wt:
            base = per_class.get("__ink__", [])
            override = opn.assign_openings(base, openings, wt) if openings else {}
            if opts.openings:
                stairs = opn.detect_stairs(base, mm_px, scale, wt)
                stairs += opn.detect_fan_stairs(base, mm_px, scale, wt, exclude={i for sd in stairs for i in sd["idx"]}
                                                | set(override))
                stairs += opn.detect_arc_stairs(base, mm_px, scale, wt, exclude={i for sd in stairs for i in sd["idx"]}
                                                | set(override))
                for sdict in stairs:
                    for i in sdict["idx"]:
                        override[i] = opn.STAIR_LAYER
        unread = set()
        if lot == "arc" and per_class.get("__ink__"):
            base_ = per_class["__ink__"]
            cen, siz = [], []
            for pts_, cl_, wq_ in base_:
                q_ = np.asarray(pts_, float)
                ext_ = q_.max(0) - q_.min(0)
                cen.append(q_.mean(0)); siz.append(float(ext_.max()))
            cen, siz = np.array(cen), np.array(siz)
            lim = 5.0 / mm_px                                    # lettre : ≤ 5 mm papier
            small_ = siz <= lim
            if small_.sum() >= 3:
                from scipy.spatial import cKDTree
                tr_ = cKDTree(cen[small_])
                idx_small = np.nonzero(small_)[0]
                for jj, ii in enumerate(idx_small):
                    nb_ = tr_.query_ball_point(cen[ii], max(3.0 * siz[ii], 0.8 * lim))
                    if len(nb_) >= 3 and not (wt and base_[ii][2] >= 0.75 * wt):
                        unread.add(int(ii))
        if cote_idx:
            base = per_class.get("__ink__", [])
            for pi_, phi in cote_idx.items():
                override[pi_] = dimmod.COTE_LAYER
                # traits d'attache (repères de cote) : petits tracés fins dont une extrémité touche la ligne de cote
                a_, b_ = np.asarray(base[pi_][0][0], float), np.asarray(base[pi_][0][-1], float)
                thp = phr[phi].get("th", phr[phi]["h"])
                for j, (q, cl_, wq) in enumerate(base):
                    if j in override or cl_ or (wt and wq >= 0.75 * wt):
                        continue
                    q = np.asarray(q, float)
                    Lq = float(np.linalg.norm(q[-1] - q[0]))
                    if Lq > 4.0 * thp:
                        continue
                    if min(np.linalg.norm(q[e] - c_) for e in (0, -1) for c_ in (a_, b_)) <= 1.2 * thp:
                        override[j] = dimmod.COTE_LAYER
        for cname, plist in per_class.items():
            for pi, (pts, closed, wpx) in enumerate(plist):
                world = np.stack([pts[:, 0] * k, (H - pts[:, 1]) * k], axis=1)
                if lot == "arc" and cname == "__ink__" and pi in override:
                    polys.append(Poly(world, override[pi], closed, wpx * k, "arc"))
                    continue
                if lot == "arc" and cname == "__ink__" and pi in unread:
                    polys.append(Poly(world, "Z-TEXTE-NON-LU", closed, wpx * k, "arc"))     # lettres que l'OCR n'a pas su lire
                    continue
                if lot == "arc":
                    layer = DISCIPLINES["arc"]["default"]
                elif not split:
                    layer = DISCIPLINES.get(lot, DISCIPLINES["autre"])["default"] if lot != "color" \
                        else DISCIPLINES["autre"]["default"]
                elif cname == "noir":
                    backdrop.append(Poly(world, "Z-FOND-ARCHI", closed, wpx * k, "autre"))
                    continue
                elif lot == "color" or opts.legend:
                    layer = legend.get(cname, "AUTRE-NON-CLASSE")
                else:
                    layer = DISCIPLINES.get(lot, DISCIPLINES["autre"])["default"]
                polys.append(Poly(world, layer, closed, wpx * k, disc_of_layer(layer)))
        # fenêtres dessinées sans interrompre le mur (double trait) : le mur est coupé à leur droit
        cutrects = [(o["rect"][0] * k, (H - o["rect"][3]) * k, o["rect"][2] * k, (H - o["rect"][1]) * k)
                    for o in openings if o.get("cut")]
        if cutrects:
            newp = []
            for p in polys:
                if p.layer.startswith("ARC-MURS"):
                    for q in opn.cut_polyline(p.pts, p.closed, cutrects):
                        newp.append(Poly(q, p.layer, False, p.width, p.disc))
                else:
                    newp.append(p)
            polys = newp
        # marqueurs : trait en travers de chaque ouverture, emprise de chaque escalier
        for o in openings:
            x0, y0, x1, y1 = o["rect"]
            if o["orient"] == "h":
                a, b = (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)
            else:
                a, b = ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)
            wpts = np.array([[a[0] * k, (H - a[1]) * k], [b[0] * k, (H - b[1]) * k]])
            polys.append(Poly(wpts, opn.WINDOW_LAYER if o["kind"] == "fenetre" else opn.DOOR_LAYER, False,
                              (wt or 0.0) * k, "arc"))
        for sdict in stairs:
            q = np.array(sdict["rect"])
            polys.append(Poly(np.stack([q[:, 0] * k, (H - q[:, 1]) * k], axis=1), opn.STAIR_LAYER, True, 0.0, "arc"))
        if lot == "arc" and polys:
            walls = [p for p in polys if p.layer in ("ARC-MURS-INT", "ARC-MURS-EXT")]
            wp = np.array([p.width / scale for p in walls]) if walls else np.array([])      # mm papier
            thr = split_walls_by_width(wp) if len(wp) else None
            if thr is not None:
                for p in walls:
                    p.layer = "ARC-MURS-EXT" if p.width / scale > thr else "ARC-MURS-INT"
            st.stats(separation_murs=f"seuil {thr:.2f} mm papier" if thr else "aucune (tout en ARC-MURS-INT)")
        counts = {}
        for p in polys:
            counts[p.layer] = counts.get(p.layer, 0) + 1
        st.stats(calques=counts, echelle=f"1:{scale:g}")
        if dims:
            st.stats(cotes=len(dims))
        if unread:
            st.stats(lettres_non_lues=len(unread))
        if openings or stairs:
            st.stats(ouvertures=len(openings), escaliers=len(stairs),
                     marches=sum(s_["marches"] for s_ in stairs))
        texts = []
        tl = text_layer(lot if lot in DISCIPLINES else "arc")
        for p_ in phr:
            if p_["conf"] < 55:
                continue
            if p_["angle"] == 0:
                px_, py_ = p_["x"], p_["y"] + p_["h"]                  # coin bas-gauche
            elif p_["angle"] == 90:                                      # lu de bas en haut : départ en bas, ligne de base à droite
                px_, py_ = p_["x"] + p_["w"], p_["y"] + p_["h"]
            else:                                                        # lu de haut en bas : départ en haut, ligne de base à gauche
                px_, py_ = p_["x"], p_["y"]
            is_cote = lot == "arc" and any(phr[i_] is p_ for i_ in cote_idx.values())
            tx = Text(np.array([px_ * k, (H - py_) * k]), p_["text"], p_["th"] * k,
                      dimmod.COTE_LAYER if is_cote else tl, p_["angle"], p_["conf"])
            if p_["conf"] < 70:
                tx.layer = "Z-TEXTE-INCERTAIN"                           # lecture douteuse : à vérifier
            texts.append(tx)
        if texts:
            st.stats(textes=len(texts))
        if backdrop:
            st.stats(fond_noir=f"{len(backdrop)} polylignes (sert au calage, retiré des lots)")
        if opts.keep_backdrop:
            polys.extend(backdrop)
    info["height_loaded"] = H
    info["world_k"] = k
    info["openings"] = openings
    info["stairs"] = len(stairs)
    return SourceResult(sid, lot, polys, backdrop, info, texts)


def pixel_to_world(info, px, py):
    """Pixel du scan NOMINAL (page entière) -> monde."""
    sc = info["scale"]
    cx, cy = info["crop_origin"]
    lx, ly = (px - cx) * sc, (py - cy) * sc
    k = info["world_k"]
    return np.array([lx * k, (info["height_loaded"] - ly) * k])


def load_dxf_source(run: Run, tag: str, src: dict) -> SourceResult:
    lot = src.get("lot", "arc")
    with run.step(tag, "Lecture du DXF") as st:
        ents, cols, skipped = read_dxf(src["file"])
        polys = []
        default = DISCIPLINES.get(lot, DISCIPLINES["autre"])["default"]
        for e in ents:
            if e["type"] != "poly":
                continue
            layer = e["layer"]
            if layer == ANCHOR_LAYER:
                continue
            layer = layer if layer in CHARTER else default
            polys.append(Poly(e["pts"], layer, e["closed"], 0.0, disc_of_layer(layer)))
        st.stats(entites=len(polys), ignorees=skipped)
    return SourceResult(src.get("id", lot), lot, polys, [], dict(file_colors=cols))


# ------------------------------------------------------------------ SVG de contrôle
def write_svg(path, polys: list[Poly], anchors=None, texts=None):
    bb = bbox(polys)
    if bb is None:
        Path(path).write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        return []
    x0, y0, x1, y1 = bb
    pad = 0.03 * max(x1 - x0, y1 - y0)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    by_layer: dict[str, list[Poly]] = {}
    for p in polys:
        by_layer.setdefault(p.layer, []).append(p)
    info = []
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{x0:.1f} {-y1:.1f} {x1 - x0:.1f} {y1 - y0:.1f}' "
             f"preserveAspectRatio='xMidYMid meet'>"]
    for name in sorted(by_layer):
        col = aci_hex(layer_color(name))
        d = []
        for p in by_layer[name]:
            pts = p.pts
            seg = "M" + "L".join(f"{x:.1f} {-y:.1f}" for x, y in pts)
            if p.closed:
                seg += "Z"
            d.append(seg)
        parts.append(f"<g id='layer-{name}' fill='none' stroke='{col}' stroke-width='1.4' "
                     f"stroke-linecap='round' stroke-linejoin='round' vector-effect='non-scaling-stroke'>"
                     f"<path d='{''.join(d)}' vector-effect='non-scaling-stroke'/></g>")
        info.append(dict(name=name, color=col, count=len(by_layer[name])))
    by_t: dict[str, list] = {}
    for t in (texts or []):
        by_t.setdefault(t.layer, []).append(t)
    for name in sorted(by_t):
        col = aci_hex(layer_color(name))
        items = []
        for t in by_t[name]:
            s_ = str(t.s).replace("&", "&amp;").replace("<", "&lt;")
            items.append(f"<text x='{t.pos[0]:.1f}' y='{-t.pos[1]:.1f}' font-size='{t.height:.1f}' "
                         f"transform='rotate({-t.angle:.2f} {t.pos[0]:.1f} {-t.pos[1]:.1f})'>{s_}</text>")
        parts.append(f"<g id='layer-{name}' fill='{col}' font-family='sans-serif'>{''.join(items)}</g>")
        info.append(dict(name=name, color=col, count=len(by_t[name])))
    parts.append("</svg>")
    Path(path).write_text("".join(parts))
    return info


def anchor_crosses(master: list[Poly]) -> list[Poly]:
    """Deux croix sur les sommets extrêmes (bas-gauche / haut-droite) du plan maître."""
    allp = np.vstack([p.pts for p in master])
    s = allp.sum(1)
    a, b = allp[np.argmin(s)], allp[np.argmax(s)]
    bb = bbox(master)
    r = 0.015 * max(bb[2] - bb[0], bb[3] - bb[1])
    out = []
    for c in (a, b):
        out.append(Poly(np.array([[c[0] - r, c[1]], [c[0] + r, c[1]]]), ANCHOR_LAYER))
        out.append(Poly(np.array([[c[0], c[1] - r], [c[0], c[1] + r]]), ANCHOR_LAYER))
    return out


def polys_to_entities(polys: list[Poly], texts: list | None = None):
    ents = [("polyline", p.layer, p.pts, p.closed) for p in polys]
    ents += [("text", t.layer, t.pos, t.height, t.s, t.angle) for t in (texts or [])]
    return ents


# ------------------------------------------------------------------ projet
LOT_ALIASES = {"mixed": "color", "mixte": "color", "couleur": "color"}


def normalize_project(project: dict, base_dir: str | Path | None = None) -> dict:
    """Accepte `files` {lot: chemin} ou `sources` [...], résout les chemins, attribue des identifiants uniques."""
    base = Path(base_dir) if base_dir else None
    out = dict(project)

    def fix_level(lv):
        lv = dict(lv)
        srcs = [dict(s) for s in lv.get("sources", [])]
        for lot, f in (lv.get("files") or {}).items():
            srcs.append(dict(file=f, lot=LOT_ALIASES.get(lot, lot)))
        seen: dict[str, int] = {}
        for s in srcs:
            s["lot"] = LOT_ALIASES.get(s.get("lot", "arc"), s.get("lot", "arc"))
            if base and not Path(s["file"]).is_absolute():
                s["file"] = str((base / s["file"]).resolve())
            sid = s.get("id") or s["lot"]
            seen[sid] = seen.get(sid, 0) + 1
            s["id"] = sid if seen[sid] == 1 else f"{sid}{seen[sid]}"
        srcs = [s for s in srcs if not s.get("skip")]
        lv["sources"] = srcs
        lv.pop("files", None)
        return lv

    if "buildings" in project:
        out["buildings"] = [dict(b, levels=[fix_level(lv) for lv in b["levels"]]) for b in project["buildings"]]
    else:
        out["levels"] = [fix_level(lv) for lv in project["levels"]]
    return out


def run_manifest(project: dict, opts: Options, outdir: str, echo=True, base_dir=None):
    """Projet à un ou plusieurs bâtiments. Un dossier de résultats (donc un run web) par bâtiment."""
    project = normalize_project(project, base_dir)
    if "buildings" not in project:
        return [run_project(project, opts, outdir, echo)]
    runs = []
    for b in project["buildings"]:
        slug = re.sub(r"[^\w+-]+", "_", b["name"]).strip("_") or "batiment"
        if echo:
            print(f"\n=== Bâtiment {b['name']} ===")
        runs.append(run_project(dict(name=b["name"], levels=b["levels"]), opts, str(Path(outdir) / slug), echo,
                                normalized=True))
    return runs


def run_project(project: dict, opts: Options, outdir: str, echo=True, normalized=False) -> Run:
    """project = {name, levels:[{name, z, height, sources:[{file, lot, page?, crop?, id?}], anchors:{id:{src,dst}}}]}"""
    if not normalized:
        project = normalize_project(project)
    name = project.get("name", "projet")
    run = Run(outdir, name, opts, echo)
    out = run.out
    try:
        levels_out = []
        meshes = []
        ref_master = None
        z_cursor = 0.0
        multi = len(project["levels"]) > 1
        for li, lv in enumerate(project["levels"]):
            lname = lv.get("name", f"N{li}")
            z0 = float(lv.get("z", z_cursor))
            h = float(lv.get("height", opts.wall_height))
            z_cursor = z0 + h
            results: list[SourceResult] = []
            for src in lv["sources"]:
                tag = f"{lname}:{src['id']}" if multi else src["id"]
                results.append(vectorize_source(run, tag, src, opts))

            mi = next((i for i, r in enumerate(results) if r.lot == "arc"), 0)
            master_r = results[mi]
            master = master_r.polys
            if not master:
                raise RuntimeError(f"Aucune géométrie extraite pour {master_r.id}")

            # calage des niveaux entre eux
            level_sim = IDENTITY
            if multi and ref_master is not None and opts.align != "none":
                with run.step(f"{lname}:calage-niveau", f"Calage du niveau {lname} sur le niveau de référence") as st:
                    level_sim, met = auto_align(ref_master, master)
                    st.stats(**met, **{k: round(v, 5) for k, v in sim_params(level_sim).items()})
                master_r.polys = transform_polys(master_r.polys, level_sim)
                master_r.texts = transform_texts(master_r.texts, level_sim)
                master = master_r.polys
            if ref_master is None:
                ref_master = master

            for i, r in enumerate(results):
                if i == mi:
                    continue
                tag = f"{lname}:{r.id}" if multi else r.id
                with run.step(tag, f"Calage sur le plan maître ({master_r.id})") as st:
                    anc = (lv.get("anchors") or {}).get(r.id) or (lv.get("anchors") or {}).get(r.lot)
                    geom = r.backdrop or r.polys
                    if anc:
                        src_w = np.array([pixel_to_world(r.info, *p) for p in anc["src"]])
                        dst_w = np.array([pixel_to_world(master_r.info, *p) for p in anc["dst"]])
                        sim = sim_compose(anchor_align(src_w, dst_w), level_sim)
                        st.stats(mode="2 points d'ancrage")
                    elif opts.align == "none" or not geom:
                        sim = level_sim
                        st.stats(mode="aucun calage")
                    else:
                        # `master` est déjà recalé sur le niveau de référence : pas de composition
                        sim, met = auto_align(master, geom)
                        st.stats(mode="automatique (Fourier-Mellin + ICP)"
                                      + (" sur le fond noir du plan" if r.backdrop else ""), **met)
                        if met["inlier_ratio"] < 0.35:
                            st.log("⚠ Calage peu fiable (peu de géométrie commune avec le plan maître) "
                                   "– fournir deux points d'ancrage dans le fichier projet.")
                    st.stats(**{k: round(v, 5) for k, v in sim_params(sim).items()})
                    r.polys = transform_polys(r.polys, sim)
                    r.texts = transform_texts(r.texts, sim)
                    r.backdrop = transform_polys(r.backdrop, sim)
                    if not r.backdrop and r.polys:
                        bb = bbox(master)
                        diag = float(np.hypot(bb[2] - bb[0], bb[3] - bb[1]))
                        tol = opts.strip_backdrop if opts.strip_backdrop > 0 else max(diag * 0.004, 30.0)
                        ov = overlap_ratio(r.polys, master, tol)
                        st.stats(recouvrement_plan_maitre=f"{ov * 100:.0f} %")
                        if opts.strip_backdrop > 0:
                            r.polys, rem = strip_backdrop(r.polys, master, tol)
                            st.stats(fond_retire=f"{rem * 100:.0f} % de la longueur (tolérance {tol:.0f} mm)",
                                     polylignes_restantes=len(r.polys))
                        elif ov > 0.3:
                            st.log(f"ℹ {ov * 100:.0f} % de ce lot recouvre le plan maître (fond d'architecture ?). "
                                   "Option --strip-backdrop pour le retirer ; attention, des réseaux posés le long "
                                   "des murs seraient aussi retirés. (Un scan en couleur évite ce problème.)")

            with run.step(f"{lname}:export" if multi else "export", "Export DXF / aperçu") as st:
                allpolys = [p for r in results for p in r.polys]
                anchors = anchor_crosses(master)
                base = f"{name}_{lname}" if multi else name
                layer_cols = {k: v[0] for k, v in CHARTER.items()}
                full = out / f"{base}.dxf"
                alltexts = [t for r in results for t in r.texts]
                write_dxf(full, polys_to_entities(allpolys + anchors, alltexts), layer_cols)
                run.add_output(f"{lname} – tous lots (DXF)" if multi else "DXF – tous lots", full)
                discs = []
                for p in allpolys:
                    if p.disc not in discs:
                        discs.append(p.disc)
                for d in discs:
                    pth = out / f"{base}_{d.upper()}.dxf"
                    write_dxf(pth, polys_to_entities([p for p in allpolys if p.disc == d] + anchors,
                                                     [t for t in alltexts if disc_of_layer(t.layer) == d]), layer_cols)
                    lab = DISCIPLINES.get(d, DISCIPLINES["autre"])["label"]
                    run.add_output(f"{lname} – {lab}" if multi else f"DXF – {lab}", pth)
                svg = out / f"{base}.svg"
                linfo = write_svg(svg, allpolys + anchors, texts=alltexts)
                st.stats(entites=len(allpolys) + len(anchors), textes=len(alltexts), calques=len(linfo))
                levels_out.append(dict(name=lname, svg=svg.name, layers=linfo, dxf=full.name))
                run.data["levels"] = levels_out
                run.save()

            if opts.three_d:
                with run.step(f"{lname}:3d" if multi else "3d", f"Extrusion 3D ({lname})" if multi else "Extrusion 3D") as st:
                    eo = ExtrudeOptions(wall_height=h, wall_thickness=opts.wall_thickness)
                    eo.elevations.update(opts.elevations)
                    mesh = build_mesh([p for p in allpolys if p.layer != "Z-FOND-ARCHI"], eo, z_base=z0, height=h)
                    meshes.append(mesh)
                    st.stats(triangles=int(sum(len(v) for v in mesh.values())), hauteur_mm=h, altitude_mm=z0)

        if opts.three_d and meshes:
            with run.step("3d", "Export 3D (OBJ / STL / DXF 3DFACE / web)") as st:
                mesh = merge_meshes(meshes)
                export_obj(out / f"{name}_3d.obj", mesh)
                export_stl(out / f"{name}_3d.stl", mesh)
                export_dxf3d(out / f"{name}_3d.dxf", mesh)
                (out / "model3d.json").write_text(json.dumps(mesh_to_json(mesh)))
                for lab, fn in (("3D – OBJ", "_3d.obj"), ("3D – STL", "_3d.stl"), ("3D – DXF (3DFACE)", "_3d.dxf")):
                    run.add_output(lab, out / f"{name}{fn}")
                run.data["model3d"] = "model3d.json"
                st.stats(triangles=int(sum(len(v) for v in mesh.values())))
        run.finish()
    except Exception as e:
        run.finish(error=f"{type(e).__name__}: {e}")
        raise
    return run
