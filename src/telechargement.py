"""Téléchargement yt-dlp : listing de playlist et récupération d'une vidéo (chemin réel + info)."""
import os

import yt_dlp

from .config import DOSSIER_BRUT
from .ffmpeg_utils import resoudre_ffmpeg


def lister_videos_playlist(url):
    """Renvoie les URLs des vidéos d'une playlist (sans téléchargement). Gère aussi une vidéo isolée."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entrees = (info or {}).get("entries")
    if not entrees:  # URL d'une vidéo isolée, pas une playlist
        return [url]
    return [f"https://www.youtube.com/watch?v={e['id']}"
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
