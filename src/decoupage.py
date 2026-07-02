"""Étape 3 : découpe et recadrage vertical 9:16 des clips via ffmpeg."""
import os

from .config import DOSSIER_SORTIE
from .ffmpeg_utils import lancer_ffmpeg


def _chemin_unique(chemin):
    """Évite d'écraser un fichier existant en suffixant _2, _3, ..."""
    if not os.path.exists(chemin):
        return chemin
    racine, ext = os.path.splitext(chemin)
    n = 2
    while os.path.exists(f"{racine}_{n}{ext}"):
        n += 1
    return f"{racine}_{n}{ext}"


def couper_et_recadrer_video(video_path, clips, base_nom=None):
    print(f"[{os.path.basename(video_path)}] Étape 3 : Rendu des clips verticaux (FFmpeg)...")
    # base_nom enrichi si fourni (ex. 2019-07-06_ufc-239_masvidal-vs-askren_ko), sinon id source.
    base = base_nom or os.path.splitext(os.path.basename(video_path))[0]

    chemins = []
    for i, clip in enumerate(clips):
        # _chemin_unique évite d'écraser un clip d'une autre vidéo au même nom enrichi.
        nom_sortie = _chemin_unique(os.path.join(DOSSIER_SORTIE, f"{base}_{i}.mp4"))
        # Filtre FFmpeg pour recadrer en format vertical 9:16 centré sur l'action de l'IA
        lancer_ffmpeg([
            "-i", video_path, "-ss", str(clip["debut"]), "-to", str(clip["fin"]),
            "-vf", f"crop=in_h*9/16:in_h:{clip['crop_x']}:0",
            "-c:v", "libx264", "-crf", "22", "-c:a", "aac", "-b:a", "128k",
            nom_sortie, "-y", "-loglevel", "error",
        ])
        print(f" -> Clip généré : {nom_sortie}")
        chemins.append(nom_sortie)
    return chemins
