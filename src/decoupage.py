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


def _filtre_cadrage(cadrage):
    """Traduit une spec de cadrage en argument de filtre ffmpeg.

    - mode 'crop' : crop 9:16 serré (largeur w à l'offset x).
    - mode 'pad'  : on élargit à l'envergure des combattants (src_w) puis on ramène en 9:16
      (cible_w × h) avec un fond flou -> les deux combattants restent visibles.
    """
    h = cadrage["h"]
    if cadrage["mode"] == "crop":
        return "-vf", f"crop={cadrage['w']}:{h}:{cadrage['x']}:0"
    cw = cadrage["cible_w"]
    graphe = (f"[0:v]crop={cadrage['src_w']}:{h}:{cadrage['x']}:0,split=2[bg][fg];"
              f"[bg]scale={cw}:{h}:force_original_aspect_ratio=increase,crop={cw}:{h},"
              f"gblur=sigma=20[b];[fg]scale={cw}:{h}:force_original_aspect_ratio=decrease[f];"
              f"[b][f]overlay=(W-w)/2:(H-h)/2")
    return "-filter_complex", graphe


def couper_et_recadrer_video(video_path, clips, base_nom=None):
    print(f"[{os.path.basename(video_path)}] Étape 3 : Rendu des clips verticaux (FFmpeg)...")
    # base_nom enrichi si fourni (ex. 2019-07-06_ufc-239_masvidal-vs-askren_ko), sinon id source.
    base = base_nom or os.path.splitext(os.path.basename(video_path))[0]

    chemins = []
    for i, clip in enumerate(clips):
        # _chemin_unique évite d'écraser un clip d'une autre vidéo au même nom enrichi.
        nom_sortie = _chemin_unique(os.path.join(DOSSIER_SORTIE, f"{base}_{i}.mp4"))
        flag, valeur = _filtre_cadrage(clip["cadrage"])  # crop serré ou zoom + fond flou
        lancer_ffmpeg([
            "-i", video_path, "-ss", str(clip["debut"]), "-to", str(clip["fin"]),
            flag, valeur,
            "-c:v", "libx264", "-crf", "22", "-c:a", "aac", "-b:a", "128k",
            nom_sortie, "-y", "-loglevel", "error",
        ])
        print(f" -> Clip généré : {nom_sortie}")
        chemins.append(nom_sortie)
    return chemins
