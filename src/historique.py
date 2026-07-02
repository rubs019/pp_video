"""Registre des vidéos déjà traitées — évite de re-télécharger/re-traiter les mêmes vidéos.

Le registre (JSON indexé par ID YouTube) vit dans `highlights_prets/` et reste lisible :
    { "cUXVXvRUllQ": {"date_traitement": "2026-07-02T14:30:00", "titre": "...", "nb_clips": 2} }
Une vidéo y est ajoutée dès qu'elle a été analysée jusqu'au bout (même sans clip produit).
"""
import json
import os
from datetime import datetime

from .config import FICHIER_HISTORIQUE, MANIFESTE_JSON


def _amorcer_depuis_manifeste():
    """1er usage : initialise le registre avec les video_id déjà présents dans le manifeste."""
    amorce = {}
    if not os.path.exists(MANIFESTE_JSON):
        return amorce
    try:
        with open(MANIFESTE_JSON, encoding="utf-8") as f:
            lignes = json.load(f)
    except (json.JSONDecodeError, OSError):
        return amorce
    for ligne in lignes or []:
        vid = (ligne or {}).get("video_id")
        if vid and vid not in amorce:
            amorce[vid] = {"date_traitement": None, "titre": "",
                           "nb_clips": 0, "source": "amorce_manifeste"}
    return amorce


def charger_historique():
    """Charge le registre. Absent -> amorçage depuis le manifeste. Corrompu -> vide."""
    if not os.path.exists(FICHIER_HISTORIQUE):
        return _amorcer_depuis_manifeste()
    try:
        with open(FICHIER_HISTORIQUE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def est_deja_traitee(video_id, historique):
    """Vrai si la vidéo figure déjà dans le registre."""
    return bool(video_id) and video_id in historique


def marquer_traitee(video_id, titre="", nb_clips=0):
    """Ajoute/actualise une entrée (load-update-write) et renvoie le registre à jour."""
    historique = charger_historique()
    if not video_id:
        return historique
    historique[video_id] = {
        "date_traitement": datetime.now().isoformat(timespec="seconds"),
        "titre": titre or "",
        "nb_clips": nb_clips,
    }
    with open(FICHIER_HISTORIQUE, "w", encoding="utf-8") as f:
        json.dump(historique, f, ensure_ascii=False, indent=2)
    return historique
