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


def _filtrer_combattants(boites, largeur, hauteur):
    """Isole les combattants : bande centrale + plus grandes boîtes (écarte officiels/public aux bords)."""
    if len(boites) <= 1:
        return list(boites)

    def aire(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    aire_max = max(aire(b) for b in boites)
    # Centre dans la bande centrale (15–85 %) ET taille comparable à la plus grande boîte.
    centraux = [b for b in boites
                if 150 <= (b[0] + b[2]) / 2 <= 850 and aire(b) >= 0.30 * aire_max]
    candidats = centraux or boites
    return sorted(candidats, key=aire, reverse=True)[:2]  # au plus 2 combattants


def _calculer_cadrage(boites, largeur, hauteur):
    """Spec de cadrage 9:16 : crop serré si les combattants tiennent, sinon zoom arrière + fond flou."""
    crop_w = int(round(hauteur * RATIO_VERTICAL)) // 2 * 2  # largeur paire (contrainte yuv420)
    xs1 = [b[0] / 1000 * largeur for b in boites]
    xs2 = [b[2] / 1000 * largeur for b in boites]
    minx, maxx = min(xs1), max(xs2)
    cx = (minx + maxx) / 2
    besoin = (maxx - minx) * 1.2  # envergure des combattants + 20 % de marge
    plafond = 2.2 * crop_w        # au-delà : le fond flou dézoomerait trop -> crop serré recentré
    if besoin <= crop_w + 1 or besoin > plafond:
        # Tiennent, OU trop écartés : crop serré 9:16 centré sur leur milieu (sujets lisibles).
        x = int(max(0, min(cx - crop_w / 2, largeur - crop_w)))
        return {"mode": "crop", "x": x, "w": crop_w, "h": hauteur}
    # Écartés mais raisonnables -> on élargit à leur envergure puis fond flou pour ramener en 9:16.
    src_w = int(min(largeur, besoin)) // 2 * 2
    x0 = int(max(0, min(cx - src_w / 2, largeur - src_w)))
    return {"mode": "pad", "x": x0, "src_w": src_w, "h": hauteur, "cible_w": crop_w}


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

        # Détection par frame ; on retient la frame au meilleur duo de combattants (nombre puis taille).
        combattants = []
        total = 0
        for chemin in chemins:
            boites = _detecter_boites(model, processor, config, chemin)
            os.remove(chemin)
            total += len(boites)
            filtres = _filtrer_combattants(boites, largeur, hauteur)
            aire = sum((b[2] - b[0]) * (b[3] - b[1]) for b in filtres)
            aire_ref = sum((b[2] - b[0]) * (b[3] - b[1]) for b in combattants)
            if (len(filtres), aire) > (len(combattants), aire_ref):
                combattants = filtres

        if combattants:
            # Action confirmée par la vision : cadrage adaptatif centré sur les combattants.
            cadrage = _calculer_cadrage(combattants, largeur, hauteur)
            clips_valides.append({"debut": debut, "fin": fin, "cadrage": cadrage})
            print(f"    Segment {debut}-{fin}s validé ({total} détection(s), "
                  f"{len(combattants)} combattant(s), cadrage={cadrage['mode']}).")
        else:
            # Aucune personne détectée : probable faux positif audio, on écarte le segment.
            print(f"    Segment {debut}-{fin}s rejeté (aucune détection visuelle).")

    _liberer_memoire_mlx()
    print(f" -> {len(clips_valides)}/{len(candidats)} segment(s) validé(s) visuellement.")
    return clips_valides
