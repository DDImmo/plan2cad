"""Garde-fous pour des fichiers NON FIABLES (plans reçus par mail, manifestes téléchargés, dossiers partagés).

* chemins d'un manifeste confinés à des racines autorisées (plus de `/etc/passwd` ni de `../../` dans un manifest.json) ;
* dossier de sortie : jamais la racine ni un dossier système ; noms sûrs dans les fichiers ;
* limites : taille de fichier, nombre de pixels (anti « bombe de décompression »), pages de PDF, côté de travail ;
* limites de ressources du processus (mémoire, temps CPU) et exécution du décodage dans un SOUS-PROCESSUS limité.
Les limites se règlent par variables d'environnement (PLAN2CAD_MAX_PIXELS, PLAN2CAD_MAX_FILE_MB, PLAN2CAD_MAX_PDF_PAGES).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

MAX_PIXELS = int(float(os.environ.get("PLAN2CAD_MAX_PIXELS", 300e6)))        # un A0 à 400 dpi ≈ 250 Mpx
MAX_FILE_MB = float(os.environ.get("PLAN2CAD_MAX_FILE_MB", 500))
MAX_PDF_PAGES = int(os.environ.get("PLAN2CAD_MAX_PDF_PAGES", 300))
MAX_SIDE_PX = 20000                                                           # côté de travail maximal
ALLOWED_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf", ".dxf"}
FORBIDDEN_OUT = {"/", "/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/dev", "/proc", "/sys", "/var",
                 "/root", "/home", "/opt", "/srv", "/mnt", "/tmp", "/System", "/Library", "/Applications", "/Users",
                 "/Windows", "/Program Files", "/Program Files (x86)"}


class UnsafeInput(ValueError):
    """Entrée refusée par un garde-fou (message en français, destiné à l'utilisateur)."""


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def safe_name(s, default: str = "x", maxlen: int = 80) -> str:
    """Nom utilisable dans un nom de fichier : ni séparateur, ni `..`, ni caractère de contrôle."""
    s = re.sub(r"[^\w+\-. ]+", "_", str(s), flags=re.UNICODE).strip(" ._")
    s = re.sub(r"\.{2,}", ".", s)
    return (s[:maxlen] or default)


def check_input_file(path, allowed_roots=None) -> Path:
    """Fichier source : existe, extension reconnue, taille raisonnable, et (si `allowed_roots`) sous une racine autorisée."""
    p = Path(path)
    try:
        rp = p.resolve(strict=True)
    except (OSError, RuntimeError):
        raise UnsafeInput(f"Fichier introuvable : {path}")
    if not rp.is_file():
        raise UnsafeInput(f"Ce n'est pas un fichier : {path}")
    if rp.suffix.lower() not in ALLOWED_EXT:
        raise UnsafeInput(f"Type de fichier non pris en charge : {rp.name}")
    if rp.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        raise UnsafeInput(f"Fichier trop volumineux ({rp.stat().st_size / 1e6:.0f} Mo > {MAX_FILE_MB:.0f} Mo) : {rp.name}")
    if allowed_roots is not None:
        roots = [Path(r).resolve() for r in allowed_roots]
        if not any(_inside(rp, r) for r in roots):
            raise UnsafeInput(
                f"Chemin hors des dossiers autorisés : {path}\n"
                f"  (autorisés : {', '.join(str(r) for r in roots)} ; ajouter --allow-dir DOSSIER si ce chemin est voulu)")
    return rp


def check_outdir(outdir) -> Path:
    """Dossier de sortie : jamais la racine, le dossier personnel ou un dossier système (les résultats y créent / remplacent
    des fichiers : steps/, run.json, *.dxf)."""
    out = Path(outdir).expanduser().resolve()
    home = Path.home().resolve()
    if str(out) in FORBIDDEN_OUT or out == home or out.parent == out or str(out).rstrip("\\/") in ("C:", "D:"):
        raise UnsafeInput(f"Dossier de sortie refusé (trop général ou système) : {out}")
    return out


def check_pixels(w: int, h: int, what: str = "image"):
    if w * h > MAX_PIXELS:
        raise UnsafeInput(f"{what} trop grande : {w}×{h} = {w * h / 1e6:.0f} Mpx > {MAX_PIXELS / 1e6:.0f} Mpx "
                          f"(PLAN2CAD_MAX_PIXELS pour relever la limite)")


def check_image_header(path):
    """Lit seulement l'en-tête (PIL ne décode rien) : refuse une bombe de décompression avant tout décodage."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            check_pixels(*im.size, what=f"Image {Path(path).name}")
            n = getattr(im, "n_frames", 1)
            if n > MAX_PDF_PAGES:
                raise UnsafeInput(f"Trop de pages/images dans {Path(path).name} ({n} > {MAX_PDF_PAGES})")
    except UnsafeInput:
        raise
    except Exception as e:
        raise UnsafeInput(f"Image illisible ({Path(path).name}) : {type(e).__name__}")


def clamp_side(max_px: int) -> int:
    return int(min(max(256, max_px), MAX_SIDE_PX))


# ----------------------------------------------------------------- ressources
def apply_limits(mem_mb: float | None = None, cpu_s: float | None = None):
    """Limites du processus courant (Unix) : espace d'adressage et temps CPU. Sans effet sous Windows."""
    try:
        import resource
    except ImportError:
        return False
    if mem_mb:
        b = int(mem_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))
    if cpu_s:
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 5))
    return True


def _child(conn, fn, args, kwargs, mem_mb, cpu_s):
    try:
        apply_limits(mem_mb, cpu_s)
        conn.send(("ok", fn(*args, **kwargs)))
    except BaseException as e:                                    # MemoryError compris
        try:
            conn.send(("err", f"{type(e).__name__}: {e}"[:500]))
        except Exception:
            pass
    finally:
        conn.close()


def run_isolated(fn, *args, mem_mb: float | None = 4096, cpu_s: float | None = 600, timeout_s: float = 900, **kwargs):
    """Exécute `fn` (fonction de module, arguments et résultat picklables) dans un sous-processus à mémoire / CPU / durée
    bornées : un PDF ou une image piégés ne peuvent ni saturer la machine ni figer l'appelant."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, fn, args, kwargs, mem_mb, cpu_s), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout_s):
            raise UnsafeInput(f"Traitement interrompu : plus de {timeout_s:.0f} s (fichier piégé ou trop complexe ?)")
        try:
            kind, val = parent.recv()
        except EOFError:
            raise UnsafeInput("Le sous-processus a été arrêté (mémoire ou temps CPU dépassé ?)")
    finally:
        if p.is_alive():
            p.kill()
        p.join(5)
    if kind == "err":
        raise UnsafeInput(f"Lecture du fichier impossible : {val}")
    return val


def isolate_enabled() -> bool:
    return os.environ.get("PLAN2CAD_ISOLATE", "") not in ("", "0", "false", "no")
