"""Résolution robuste de ffmpeg (indépendante du PATH du shell) et exécution bruyante."""
import os
import shutil
import subprocess

_ffmpeg_bin = None


def resoudre_ffmpeg():
    """Localise le binaire ffmpeg une fois pour toutes : PATH, puis emplacements Homebrew usuels."""
    global _ffmpeg_bin
    if _ffmpeg_bin is None:
        chemin = shutil.which("ffmpeg")
        if not chemin:
            for repli in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
                if os.path.exists(repli):
                    chemin = repli
                    break
        if not chemin:
            raise RuntimeError(
                "ffmpeg introuvable (ni dans le PATH, ni dans /opt/homebrew/bin ou "
                "/usr/local/bin). Installez-le (brew install ffmpeg) ou ajoutez son dossier au PATH."
            )
        _ffmpeg_bin = chemin
    return _ffmpeg_bin


def lancer_ffmpeg(arguments):
    """Exécute ffmpeg via le binaire résolu (sans shell) et échoue bruyamment si le code retour ≠ 0."""
    resultat = subprocess.run(
        [resoudre_ffmpeg(), *arguments],
        capture_output=True, text=True,
    )
    if resultat.returncode != 0:
        raise RuntimeError(
            f"ffmpeg a échoué (code {resultat.returncode}) : {resultat.stderr.strip()[-500:]}"
        )
