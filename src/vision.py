"""Étape 2 : validation visuelle via LocateAnything-3B (MLX) et calcul du recadrage dynamique."""
import gc
import os
import re

import cv2

from .config import (DOSSIER_BRUT, FRAMES_PAR_SEGMENT, INTERVALLE_SUIVI, LISSAGE_FENETRE,
                     MODELE_VISION, PAN_VITESSE_MAX, PROMPT_DETECTION, RATIO_VERTICAL,
                     RECADRAGE_DYNAMIQUE, SEUIL_PAN_FIT)

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


def _echantillonner_trajectoire(video_path, debut, fin, n):
    """Comme _echantillonner_frames mais conserve l'instant : renvoie (points, largeur, hauteur)
    avec points = [(chemin_jpg, t_relatif)], t_relatif = instant - debut (0..durée)."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return [], 0, 0
    largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    points = []
    for k in range(n):
        instant = debut + (fin - debut) * (k + 1) / (n + 1)
        capture.set(cv2.CAP_PROP_POS_MSEC, instant * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        chemin = os.path.join(DOSSIER_BRUT, f"_traj_{int(instant)}_{k}.jpg")
        cv2.imwrite(chemin, frame)
        points.append((chemin, instant - debut))
    capture.release()
    return points, largeur, hauteur


def _centre_span(boites, largeur):
    """(cx, span) en pixels : cx = milieu de l'envergure des combattants, span = largeur occupée."""
    xs1 = [b[0] / 1000 * largeur for b in boites]
    xs2 = [b[2] / 1000 * largeur for b in boites]
    minx, maxx = min(xs1), max(xs2)
    return (minx + maxx) / 2, (maxx - minx)


def _remplir_trous(vals):
    """Forward-fill puis back-fill des None (keyframes sans détection). None si tout est None."""
    if all(v is None for v in vals):
        return None
    out = list(vals)
    dernier = None
    for i, v in enumerate(out):
        if v is None:
            out[i] = dernier
        else:
            dernier = v
    suivant = None
    for i in range(len(out) - 1, -1, -1):  # comble les None de tête
        if out[i] is None:
            out[i] = suivant
        else:
            suivant = out[i]
    return out


def _lisser_trajectoire(t_list, cx_list, crop_w):
    """Moyenne glissante centrée (LISSAGE_FENETRE) puis limitation de vitesse de pano (anti à-coups)."""
    n = len(cx_list)
    demi = LISSAGE_FENETRE // 2
    liss = []
    for i in range(n):
        fenetre = cx_list[max(0, i - demi):min(n, i + demi + 1)]
        liss.append(sum(fenetre) / len(fenetre))
    v_max = PAN_VITESSE_MAX * crop_w  # px/s
    out = [liss[0]]
    for i in range(1, n):
        dmax = v_max * max(1e-3, t_list[i] - t_list[i - 1])
        delta = max(-dmax, min(liss[i] - out[-1], dmax))
        out.append(out[-1] + delta)
    return out


def _construire_expr_pan(t_list, x_list, x_max):
    """Expression ffmpeg x(t) : interpolation linéaire par morceaux, paliers aux extrémités,
    bornage dans [0, x_max] et offset pair (yuv420)."""
    expr = f"{x_list[-1]:.1f}"  # palier après la dernière keyframe
    for k in range(len(t_list) - 2, -1, -1):
        dt = t_list[k + 1] - t_list[k]
        pente = (x_list[k + 1] - x_list[k]) / dt if dt > 1e-6 else 0.0
        seg = f"{x_list[k]:.1f}+({pente:.4f})*(t-{t_list[k]:.3f})"
        expr = f"if(lt(t,{t_list[k + 1]:.3f}),{seg},{expr})"
    expr = f"if(lt(t,{t_list[0]:.3f}),{x_list[0]:.1f},{expr})"  # palier avant la 1re keyframe
    return f"2*floor(clip({expr},0,{x_max})/2)"  # bornage + offset pair


def _cadrage_statique(model, processor, config, video_path, debut, fin):
    """Cadrage figé (comportement historique) : renvoie (cadrage, info) ou (None, raison)."""
    chemins, largeur, hauteur = _echantillonner_frames(video_path, debut, fin, FRAMES_PAR_SEGMENT)
    if not chemins:
        return None, "frames illisibles"
    combattants, total = [], 0
    for chemin in chemins:
        boites = _detecter_boites(model, processor, config, chemin)
        os.remove(chemin)
        total += len(boites)
        filtres = _filtrer_combattants(boites, largeur, hauteur)
        aire = sum((b[2] - b[0]) * (b[3] - b[1]) for b in filtres)
        aire_ref = sum((b[2] - b[0]) * (b[3] - b[1]) for b in combattants)
        if (len(filtres), aire) > (len(combattants), aire_ref):
            combattants = filtres
    if not combattants:
        return None, "aucune détection visuelle"
    cadrage = _calculer_cadrage(combattants, largeur, hauteur)
    return cadrage, f"{total} détection(s), {len(combattants)} combattant(s), cadrage={cadrage['mode']}"


def _cadrage_pan(model, processor, config, video_path, debut, fin):
    """Panoramique fluide : détections à intervalles -> trajectoire lissée -> expression x(t).
    Renvoie (cadrage, info) ou (None, raison). Repli statique si les combattants sont trop écartés."""
    n = max(2, round((fin - debut) / INTERVALLE_SUIVI))
    points, largeur, hauteur = _echantillonner_trajectoire(video_path, debut, fin, n)
    if not points:
        return None, "frames illisibles"
    crop_w = int(round(hauteur * RATIO_VERTICAL)) // 2 * 2

    t_list, cx_raw, spans = [], [], []
    meilleur, meilleur_score = [], (-1, -1)
    for chemin, t_rel in points:
        boites = _detecter_boites(model, processor, config, chemin)
        os.remove(chemin)
        filtres = _filtrer_combattants(boites, largeur, hauteur)
        t_list.append(t_rel)
        if filtres:
            cx, span = _centre_span(filtres, largeur)
            cx_raw.append(cx)
            spans.append(span)
            aire = sum((b[2] - b[0]) * (b[3] - b[1]) for b in filtres)
            if (len(filtres), aire) > meilleur_score:
                meilleur_score, meilleur = (len(filtres), aire), filtres
        else:
            cx_raw.append(None)
            spans.append(None)

    if not any(c is not None for c in cx_raw):
        return None, "aucune détection visuelle"

    spans_ok = [s for s in spans if s is not None]
    fit = sum(1 for s in spans_ok if s * 1.2 <= crop_w) / len(spans_ok)
    nb = len(spans_ok)
    if nb >= 2 and fit >= SEUIL_PAN_FIT:
        cx_liss = _lisser_trajectoire(t_list, _remplir_trous(cx_raw), crop_w)
        x_list = [int(max(0, min(round(c - crop_w / 2), largeur - crop_w))) for c in cx_liss]
        expr = _construire_expr_pan(t_list, x_list, largeur - crop_w)
        return {"mode": "pan", "w": crop_w, "h": hauteur, "expr": expr}, \
            f"{nb} keyframe(s) suivie(s), cadrage=pan"
    cadrage = _calculer_cadrage(meilleur, largeur, hauteur)  # repli statique (crop/pad)
    return cadrage, f"trop écartés (fit={fit:.0%}), repli cadrage={cadrage['mode']}"


def verifier_et_cadrer_locate_anything(video_path, candidats):
    mode = "panoramique" if RECADRAGE_DYNAMIQUE else "statique"
    print(f"[{os.path.basename(video_path)}] Étape 2 : Validation visuelle (LocateAnything-3B, {mode})...")
    model, processor, config = _charger_modele_vision()
    clips_valides = []

    for debut, fin in candidats:
        if RECADRAGE_DYNAMIQUE:
            cadrage, info = _cadrage_pan(model, processor, config, video_path, debut, fin)
        else:
            cadrage, info = _cadrage_statique(model, processor, config, video_path, debut, fin)
        if cadrage is None:
            print(f"    Segment {debut}-{fin}s rejeté ({info}).")
            continue
        clips_valides.append({"debut": debut, "fin": fin, "cadrage": cadrage})
        print(f"    Segment {debut}-{fin}s validé ({info}).")

    _liberer_memoire_mlx()
    print(f" -> {len(clips_valides)}/{len(candidats)} segment(s) validé(s) visuellement.")
    return clips_valides
