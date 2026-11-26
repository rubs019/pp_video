"""Téléchargement yt-dlp : listing de playlist et récupération d'une vidéo (chemin réel + info)."""
import os
import re
import shutil

import yt_dlp

from .config import DOSSIER_BRUT, FICHIER_COOKIES
from .ffmpeg_utils import resoudre_ffmpeg


def _trouver_binaire(nom):
    """Localise un binaire dans le PATH ou les emplacements Homebrew/macOS usuels."""
    chemin = shutil.which(nom)
    if chemin:
        return chemin
    for dossier in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidat = os.path.join(dossier, nom)
        if os.path.exists(candidat):
            return candidat
    return None


def _options_communes():
    """Options yt-dlp partagées : cookies.txt (si présent) + runtime JS (deno/node pour EJS)."""
    opts = {}
    if FICHIER_COOKIES and os.path.exists(FICHIER_COOKIES):
        opts["cookiefile"] = FICHIER_COOKIES
    runtimes = {}
    for nom in ("deno", "node"):
        chemin = _trouver_binaire(nom)
        if chemin:
            runtimes[nom] = {"path": chemin}
    if runtimes:
        opts["js_runtimes"] = runtimes
    return opts


def _extraire_id_youtube(url):
    """Extrait l'ID (11 caractères) d'une URL YouTube ; None si introuvable."""
    if not url:
        return None
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def lister_videos_playlist(url):
    """Renvoie des paires (video_id, url), sans téléchargement. Gère aussi une vidéo isolée.

    video_id peut être None pour une URL isolée non standard (le saut pré-téléchargement
    est alors ignoré ; la vidéo sera marquée après coup via son id réel).
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        **_options_communes(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entrees = (info or {}).get("entries")
    if not entrees:  # URL d'une vidéo isolée, pas une playlist
        return [(_extraire_id_youtube(url), url)]
    return [(e["id"], f"https://www.youtube.com/watch?v={e['id']}")
            for e in entrees if e and e.get("id")]


def telecharger_video(url):
    """Télécharge une vidéo (720p max) et renvoie (chemin, info), ou (None, None) si échec."""
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": os.path.join(DOSSIER_BRUT, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": os.path.dirname(resoudre_ffmpeg()),  # le merge yt-dlp a besoin de ffmpeg (hors PATH)
        "ignoreerrors": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        },
        **_options_communes(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        return None, None
    try:
        chemin = info["requested_downloads"][0]["filepath"]  # chemin réel, sans ambiguïté glob
    except (KeyError, IndexError, TypeError):
        chemin = os.path.splitext(ydl.prepare_filename(info))[0] + ".mp4"
        if not os.path.exists(chemin):
            chemin = None
    return chemin, info
