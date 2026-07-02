"""Étape 2 : validation visuelle via LocateAnything-3B (MLX) et calcul du recadrage dynamique."""
import gc
import os
import re

import cv2

from .config import (DOSSIER_BRUT, FRAMES_PAR_SEGMENT, MODELE_VISION,
                     PROMPT_DETECTION, RATIO_VERTICAL)

# --- Modèle de vision chargé une seule fois (coûteux : ~3 Go en mémoire unifiée) ---
_modele_vision = None


def _charger_modele_vision():
    """Charge LocateAnything-3B (MLX) et le met en cache pour tout le run."""
    global _modele_vision
    if _modele_vision is None:
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise RuntimeError(
                "mlx-vlm est requis pour l'étape 2 : pip install mlx-vlm "
                "(le support LocateAnything peut nécessiter une branche dédiée)."
            ) from e
        print(f"    Chargement de {MODELE_VISION} (une seule fois, patience)...")
        model, processor = load(MODELE_VISION)
        config = load_config(MODELE_VISION)
        _modele_vision = (model, processor, config)
    return _modele_vision


def _liberer_memoire_mlx():
    """Vide la mémoire unifiée du Mac (M4) après inférence."""
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _detecter_boites(model, processor, config, chemin_image):
    """Lance le grounding sur une image et renvoie les boîtes [(x1,y1,x2,y2)] en 0-1000."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, config, PROMPT_DETECTION, num_images=1)
    sortie = generate(
        model, processor, prompt, [chemin_image],
        max_tokens=256, temperature=0.0, verbose=False,
    )
    texte = getattr(sortie, "text", sortie)  # certaines versions renvoient un objet
    # Format LocateAnything : <ref>...</ref><box><x1><y1><x2><y2></box>, coords 0-1000
    return [tuple(map(int, b)) for b in re.findall(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", texte)]


def _echantillonner_frames(video_path, debut, fin, n):
    """Extrait n frames réparties sur le segment. Renvoie (chemins_temp, largeur, hauteur)."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return [], 0, 0
    largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chemins = []
    for k in range(n):
        instant = debut + (fin - debut) * (k + 1) / (n + 1)  # frames réparties dans le segment
        capture.set(cv2.CAP_PROP_POS_MSEC, instant * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        chemin = os.path.join(DOSSIER_BRUT, f"_frame_{int(instant)}_{k}.jpg")
        cv2.imwrite(chemin, frame)
        chemins.append(chemin)
    capture.release()
    return chemins, largeur, hauteur


def _calculer_crop_x(boites, largeur, hauteur):
    """Centre le crop 9:16 sur le barycentre horizontal des boxeurs détectés."""
    largeur_crop = hauteur * RATIO_VERTICAL
    # Centre X moyen des boîtes, ramené des coords 0-1000 vers les pixels réels.
    centres = [((x1 + x2) / 2) / 1000 * largeur for x1, y1, x2, y2 in boites]
    centre_px = sum(centres) / len(centres)
    crop_x = centre_px - largeur_crop / 2
    # On borne pour rester dans le cadre d'origine (repli à 0 si source déjà plus étroite que 9:16).
    crop_x = max(0, min(crop_x, largeur - largeur_crop))
    return int(crop_x)


def verifier_et_cadrer_locate_anything(video_path, candidats):
    print(f"[{os.path.basename(video_path)}] Étape 2 : Validation visuelle (LocateAnything-3B)...")
    model, processor, config = _charger_modele_vision()
    clips_valides = []

    for debut, fin in candidats:
        chemins, largeur, hauteur = _echantillonner_frames(
            video_path, debut, fin, FRAMES_PAR_SEGMENT)
        if not chemins:
            print(f"    Segment {debut}-{fin}s : frames illisibles, ignoré.")
            continue

        # On agrège les détections de toutes les frames échantillonnées du segment.
        boites = []
        for chemin in chemins:
            boites.extend(_detecter_boites(model, processor, config, chemin))
            os.remove(chemin)

        if boites:
            # Action confirmée par la vision : recadrage dynamique sur les boxeurs.
            crop_x = _calculer_crop_x(boites, largeur, hauteur)
            clips_valides.append({"debut": debut, "fin": fin, "crop_x": crop_x})
            print(f"    Segment {debut}-{fin}s validé ({len(boites)} détection(s)), crop_x={crop_x}.")
        else:
            # Aucun boxeur détecté : probable faux positif audio, on écarte le segment.
            print(f"    Segment {debut}-{fin}s rejeté (aucune détection visuelle).")

    _liberer_memoire_mlx()
    print(f" -> {len(clips_valides)}/{len(candidats)} segment(s) validé(s) visuellement.")
    return clips_valides
