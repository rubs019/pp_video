"""Étape 1 : détection des moments forts par énergie RMS et seuil de percentile adaptatif."""
import os

import numpy as np
from scipy.io import wavfile

from .config import DUREE_HIGHLIGHT, MARGE_MIN_DB, PERCENTILE_PIC
from .ffmpeg_utils import lancer_ffmpeg


def analyser_audio_fast(video_path):
    print(f"[{os.path.basename(video_path)}] Étape 1 : Analyse audio rapide...")
    audio_temp = os.path.splitext(video_path)[0] + ".wav"
    lancer_ffmpeg(["-i", video_path, "-ab", "128k", "-ac", "1", "-ar", "22050",
                   "-vn", audio_temp, "-y", "-loglevel", "error"])

    if not os.path.exists(audio_temp):
        return []

    samplerate, data = wavfile.read(audio_temp)
    os.remove(audio_temp)

    # --- Normalisation du signal ---
    # Passage en float64 AVANT toute mise au carré : évite l'overflow des entiers (int16²).
    data = data.astype(np.float64)
    # Repli défensif si le fichier est stéréo malgré le -ac 1 : on moyenne les canaux.
    if data.ndim > 1:
        data = data.mean(axis=1)

    amplitude_max = np.max(np.abs(data))
    if amplitude_max == 0:
        print(" -> Signal audio muet : aucun candidat.")
        return []
    data /= amplitude_max

    # --- Énergie RMS par fenêtre ---
    FENETRE_SEC = 0.5  # Résolution temporelle de l'analyse
    taille_fenetre = int(samplerate * FENETRE_SEC)
    nb_fenetres = len(data) // taille_fenetre
    if nb_fenetres == 0:
        print(" -> Audio trop court pour l'analyse.")
        return []

    blocs = data[:nb_fenetres * taille_fenetre].reshape(nb_fenetres, taille_fenetre)
    rms = np.sqrt(np.mean(blocs ** 2, axis=1))
    rms_db = 20 * np.log10(rms + 1e-10)  # +epsilon pour éviter log10(0)

    # --- Seuil adaptatif par percentile ---
    # Un seuil "médiane + N dB" fixe suppose une dynamique constante : il casse sur l'audio
    # de diffusion compressé (dynamique ~8 dB observée). On retient donc les fenêtres du top
    # percentile — auto-adaptatif à la dynamique réelle — avec un plancher de marge anti faux positifs.
    niveau_ambiant = np.median(rms_db)
    seuil = max(np.percentile(rms_db, PERCENTILE_PIC), niveau_ambiant + MARGE_MIN_DB)
    fenetres_chaudes = np.where(rms_db > seuil)[0]

    if len(fenetres_chaudes) == 0:
        print(f" -> Aucun pic audio détecté (seuil : {seuil:.1f} dB).")
        return []

    # --- Regroupement des fenêtres chaudes en moments forts ---
    # Deux pics séparés de moins de DUREE_HIGHLIGHT relèvent du même temps fort.
    ecart_max = int(DUREE_HIGHLIGHT / FENETRE_SEC)
    clusters = []
    cluster_courant = [fenetres_chaudes[0]]
    for idx in fenetres_chaudes[1:]:
        if idx - cluster_courant[-1] <= ecart_max:
            cluster_courant.append(idx)
        else:
            clusters.append(cluster_courant)
            cluster_courant = [idx]
    clusters.append(cluster_courant)

    # --- Construction des segments (début/fin en secondes) ---
    duree_totale = len(data) / samplerate
    candidats = []
    for cluster in clusters:
        # Fenêtre la plus intense du cluster = cœur de l'action.
        idx_pic = cluster[int(np.argmax(rms_db[cluster]))]
        instant_pic = idx_pic * FENETRE_SEC
        # La clameur suit l'action : on garde plus de temps AVANT le pic (pré-roll 70 %).
        debut = max(0.0, instant_pic - DUREE_HIGHLIGHT * 0.7)
        fin = min(duree_totale, instant_pic + DUREE_HIGHLIGHT * 0.3)
        candidats.append((int(debut), int(fin)))

    # --- Fusion des segments qui se chevauchent ---
    candidats.sort()
    fusionnes = [candidats[0]]
    for debut, fin in candidats[1:]:
        dernier_debut, dernier_fin = fusionnes[-1]
        if debut <= dernier_fin:
            fusionnes[-1] = (dernier_debut, max(dernier_fin, fin))
        else:
            fusionnes.append((debut, fin))

    print(f" -> {len(fusionnes)} moment(s) fort(s) détecté(s) "
          f"(seuil : {seuil:.1f} dB / ambiant : {niveau_ambiant:.1f} dB).")
    return fusionnes
