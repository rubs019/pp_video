"""Orchestration : enchaînement des étapes par vidéo et boucle résiliente sur la playlist."""
import os

from .audio import analyser_audio_fast
from .config import URL_SOURCE
from .decoupage import couper_et_recadrer_video
from .enrichissement import (construire_base_nom, ecrire_legende_txt,
                             enrichir_metadonnees, maj_manifeste)
from .telechargement import lister_videos_playlist, telecharger_video
from .vision import verifier_et_cadrer_locate_anything


def traiter_video(video_path, info):
    """Enchaîne les étapes 1 → 2 → 3 → 4 sur une vidéo déjà téléchargée."""
    info = info or {}
    candidats = analyser_audio_fast(video_path)
    if not candidats:
        print(" -> Aucun moment fort audio, vidéo ignorée.")
        return
    clips = verifier_et_cadrer_locate_anything(video_path, candidats)
    if not clips:
        print(" -> Aucun clip validé visuellement.")
        return
    # Étape 4 : 1 seul appel LLM par vidéo (métadonnées communes à tous les clips).
    meta = enrichir_metadonnees(info)
    base_nom = construire_base_nom(meta, info.get("id"))
    chemins = couper_et_recadrer_video(video_path, clips, base_nom=base_nom)
    for i, chemin in enumerate(chemins):
        ecrire_legende_txt(chemin, meta, i)
    total = maj_manifeste(chemins, meta, info)
    print(f" -> {len(chemins)} clip(s) enrichis + légendes. Manifeste : {total} entrée(s).")


def executer_pipeline_complet(url_source=None):
    # URL passée en argument prioritaire ; sinon repli sur la valeur de config.
    url_source = url_source or URL_SOURCE
    print("=== DÉMARRAGE DE LA MACHINE À HIGHLIGHTS EN FLUX TENDU ===")
    videos = lister_videos_playlist(url_source)
    print(f"{len(videos)} vidéo(s) à traiter.")

    for i, url in enumerate(videos, 1):
        print(f"\n--- [{i}/{len(videos)}] {url} ---")
        video_path = None
        try:
            video_path, info = telecharger_video(url)
            if not video_path or not os.path.exists(video_path):
                print(" -> Échec du téléchargement, on passe à la suivante.")
                continue
            traiter_video(video_path, info)
        except Exception as e:
            # Flux tendu résilient : un échec isolé ne doit pas faire tomber tout le lot.
            print(f" -> ERREUR sur cette vidéo, on continue : {e}")
        finally:
            # Suppression de la source pour préserver l'espace disque du Mac.
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(" -> Source brute supprimée.")

    print("\n" + "=" * 60)
    print("=== FLUX TERMINÉ ===")
