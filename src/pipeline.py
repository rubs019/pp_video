"""Orchestration : enchaînement des étapes par vidéo et boucle résiliente sur la playlist."""
import os

from .audio import analyser_audio_fast
from .config import DEBUG_VIDEO, URL_SOURCE
from .debug_visuel import generer_videos_debug
from .decoupage import couper_et_recadrer_video
from .enrichissement import (construire_base_nom, ecrire_legende_video,
                             enrichir_metadonnees, maj_manifeste)
from .historique import charger_historique, est_deja_traitee, marquer_traitee
from .telechargement import lister_videos_playlist, telecharger_video
from .vision import verifier_et_cadrer_vision


def traiter_video(video_path, info, debug_video=False):
    """Enchaîne les étapes 1 → 2 → 3 → 4 sur une vidéo déjà téléchargée. Renvoie le nb de clips."""
    info = info or {}
    candidats = analyser_audio_fast(video_path)
    if not candidats:
        print(" -> Aucun moment fort audio, vidéo ignorée.")
        return 0
    clips = verifier_et_cadrer_vision(video_path, candidats)
    if not clips:
        print(" -> Aucun clip validé visuellement.")
        return 0
    # Étape 4 : 1 seul appel LLM par vidéo (métadonnées communes à tous les clips).
    meta = enrichir_metadonnees(info)
    base_nom = construire_base_nom(meta, info.get("id"))
    chemins = couper_et_recadrer_video(video_path, clips, base_nom=base_nom)
    if debug_video:
        generer_videos_debug(video_path, clips, chemins)
    legende_txt = ecrire_legende_video(base_nom, chemins, meta)  # un seul .txt par vidéo
    total = maj_manifeste(chemins, meta, info, legende_txt)
    print(f" -> {len(chemins)} clip(s) + 1 légende vidéo. Manifeste : {total} entrée(s).")
    return len(chemins)


def executer_pipeline_complet(url_source=None, force=False, debug_video=None):
    # URL passée en argument prioritaire ; sinon repli sur la valeur de config.
    url_source = url_source or URL_SOURCE
    debug_video = DEBUG_VIDEO if debug_video is None else debug_video
    print("=== DÉMARRAGE DE LA MACHINE À HIGHLIGHTS EN FLUX TENDU ===")
    historique = charger_historique()
    videos = lister_videos_playlist(url_source)
    print(f"{len(videos)} vidéo(s) dans la source ({len(historique)} déjà au registre).")

    for i, (video_id, url) in enumerate(videos, 1):
        print(f"\n--- [{i}/{len(videos)}] {url} ---")
        # Saut AVANT téléchargement : on n'a rien à re-faire sur une vidéo déjà analysée.
        if video_id and not force and est_deja_traitee(video_id, historique):
            print(f" -> Déjà traitée ({video_id}), ignorée. (--force pour retraiter)")
            continue

        video_path = None
        try:
            video_path, info = telecharger_video(url)
            if not video_path or not os.path.exists(video_path):
                print(" -> Échec du téléchargement, on passe à la suivante.")
                continue  # non marquée : nouvelle tentative au prochain run
            info = info or {}
            nb = traiter_video(video_path, info, debug_video=debug_video)
            # Analyse menée à son terme -> on enregistre la vidéo (même si 0 clip).
            historique = marquer_traitee(info.get("id") or video_id,
                                         info.get("title", ""), nb)
        except Exception as e:
            # Flux tendu résilient : un échec isolé ne doit pas faire tomber tout le lot.
            # Non marquée non plus -> retentée au prochain run.
            print(f" -> ERREUR sur cette vidéo, on continue : {e}")
        finally:
            # Suppression de la source pour préserver l'espace disque du Mac.
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(" -> Source brute supprimée.")

    print("\n" + "=" * 60)
    print("=== FLUX TERMINÉ ===")
