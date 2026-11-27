"""Rendu vidéo Debug Overlay / Split-Screen (--debug-video).

Génère une vidéo vitrine côte-à-côte pour chaque clip :
- À gauche : flux 16:9 original annoté (boîtes VLM, fenêtre de cadrage 9:16 active,
  bandeau HUD et courbe d'énergie audio RMS en temps réel).
- À droite : clip final 9:16 synchronisé (lu depuis le rendu FFmpeg définitif).
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np
from scipy.io import wavfile

from .config import DOSSIER_DEBUG, MODELE_VISION
from .ffmpeg_utils import lancer_ffmpeg, resoudre_ffmpeg


def _extraire_rms_clip(chemin_clip):
    """Calcule la courbe d'énergie RMS (normalisée 0..1) du clip par pas de 0,25 s."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        lancer_ffmpeg(["-i", chemin_clip, "-ac", "1", "-ar", "16000", "-vn",
                       wav_path, "-y", "-loglevel", "error"])
        sr, data = wavfile.read(wav_path)
    except Exception:
        return None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    pic = np.max(np.abs(data))
    if pic == 0:
        return None
    data /= pic

    pas = max(1, int(sr * 0.25))
    nb = len(data) // pas
    if nb < 2:
        return None
    blocs = data[:nb * pas].reshape(nb, pas)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(blocs ** 2, axis=1)) + 1e-10)
    bas, haut = float(np.percentile(rms_db, 5)), float(np.max(rms_db))
    etendue = max(1e-3, haut - bas)
    return np.clip((rms_db - bas) / etendue, 0.0, 1.0)


def _eval_pan_x(t_rel, t_list, x_list, x_max):
    """Évalue la position horizontale x(t) du panoramique (identique à _construire_expr_pan)."""
    if not t_list or not x_list:
        return 0
    if t_rel <= t_list[0]:
        val = x_list[0]
    elif t_rel >= t_list[-1]:
        val = x_list[-1]
    else:
        val = x_list[-1]
        for k in range(len(t_list) - 1):
            if t_list[k] <= t_rel <= t_list[k + 1]:
                dt = t_list[k + 1] - t_list[k]
                ratio = (t_rel - t_list[k]) / dt if dt > 1e-6 else 0.0
                val = x_list[k] + ratio * (x_list[k + 1] - x_list[k])
                break
    return int(max(0, min(x_max, val))) // 2 * 2


def _index_keyframe(t_rel, t_list):
    """Indice de la keyframe la plus proche de t_rel."""
    if not t_list:
        return None
    return min(range(len(t_list)), key=lambda i: abs(t_list[i] - t_rel))


def _dessiner_cadre_cadrage(frame, cadrage, debug, t_rel):
    """Assombrit les zones hors cadrage et trace le rectangle actif (pan / crop / pad)."""
    h, w = frame.shape[:2]
    mode = cadrage["mode"]

    if mode == "pan":
        crop_w = cadrage["w"]
        t_list = (debug or {}).get("t_list") or []
        x_list = (debug or {}).get("x_list") or []
        x0 = _eval_pan_x(t_rel, t_list, x_list, max(0, w - crop_w))
        x1 = min(w, x0 + crop_w)
        couleur = (255, 40, 220)  # magenta
        etiquette = "9:16 PAN"
    elif mode == "crop":
        x0 = max(0, cadrage["x"])
        x1 = min(w, x0 + cadrage["w"])
        couleur = (255, 40, 220)
        etiquette = "9:16 CROP"
    else:  # pad
        x0 = max(0, cadrage["x"])
        x1 = min(w, x0 + cadrage["src_w"])
        couleur = (0, 185, 255)  # orange/ambre
        etiquette = "9:16 PAD+BLUR"

    # Assombrissement léger hors de la fenêtre retenue pour visualiser le recadrage
    voile = frame.copy()
    if x0 > 0:
        voile[:, :x0] = (voile[:, :x0] * 0.45).astype(np.uint8)
    if x1 < w:
        voile[:, x1:] = (voile[:, x1:] * 0.45).astype(np.uint8)
    frame[:] = voile

    cv2.rectangle(frame, (x0, 2), (max(x0 + 1, x1 - 1), h - 3), couleur, 3)
    cv2.rectangle(frame, (x0, h - 34), (min(w, x0 + 148), h - 4), couleur, -1)
    cv2.putText(frame, etiquette, (x0 + 8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 2, cv2.LINE_AA)


def _dessiner_boites(frame, debug, t_rel):
    """Dessine les détections brutes (jaune) et combattants filtrés (vert) de la keyframe active."""
    if not debug:
        return
    t_list = debug.get("t_list") or []
    idx = _index_keyframe(t_rel, t_list)
    if idx is None:
        return
    h, w = frame.shape[:2]
    raw_boxes = (debug.get("raw_boxes") or [])[idx]
    filt_boxes = (debug.get("filtered_boxes") or [])[idx]

    for x1, y1, x2, y2 in raw_boxes:
        cv2.rectangle(frame, (int(x1 / 1000 * w), int(y1 / 1000 * h)),
                      (int(x2 / 1000 * w), int(y2 / 1000 * h)), (0, 220, 255), 1, cv2.LINE_AA)

    for j, (x1, y1, x2, y2) in enumerate(filt_boxes, 1):
        p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
        p2 = (int(x2 / 1000 * w), int(y2 / 1000 * h))
        cv2.rectangle(frame, p1, p2, (50, 245, 90), 3, cv2.LINE_AA)
        tag = f"FIGHTER {j}"
        ty = max(24, p1[1] - 8)
        cv2.rectangle(frame, (p1[0], ty - 18), (min(w, p1[0] + 92), ty + 4), (50, 245, 90), -1)
        cv2.putText(frame, tag, (p1[0] + 5, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)


def _dessiner_hud_et_audio(frame, mode, t_rel, duree, rms_norm):
    """Bandeau HUD supérieur + mini-courbe d'énergie audio RMS en bas à gauche."""
    h, w = frame.shape[:2]
    nom_modele = MODELE_VISION.split("/")[-1].replace("-Instruct-4bit", "")

    # Bandeau supérieur semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    texte = f"AI PIPELINE (MLX LOCAL)  |  {nom_modele}  |  MODE: {mode.upper()}  |  t={t_rel:04.1f}s / {duree:.0f}s"
    cv2.putText(frame, texte, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (240, 245, 255), 1, cv2.LINE_AA)

    # Mini-graphe audio RMS en bas à gauche
    if rms_norm is not None and len(rms_norm) >= 2:
        gx, gy, gw, gh = 16, h - 98, 280, 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (gx - 6, gy - 20), (gx + gw + 6, gy + gh + 6), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "AUDIO ENERGY (RMS)", (gx, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 210, 230), 1, cv2.LINE_AA)

        n = len(rms_norm)
        pts = np.array([
            [int(gx + i * gw / (n - 1)), int(gy + gh - rms_norm[i] * (gh - 6) - 3)]
            for i in range(n)
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], False, (255, 210, 60), 2, cv2.LINE_AA)

        # Curseur temporel
        prog = max(0.0, min(1.0, t_rel / max(1e-3, duree)))
        cx = int(gx + prog * gw)
        cv2.line(frame, (cx, gy), (cx, gy + gh), (255, 255, 255), 2, cv2.LINE_AA)


def _dessiner_badge_sortie(frame_out):
    """Badge discret en haut du clip final 9:16."""
    w = frame_out.shape[1]
    overlay = frame_out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (12, 14, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame_out, 0.28, 0, frame_out)
    cv2.putText(frame_out, "OUTPUT 9:16", (14, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (50, 245, 90), 2, cv2.LINE_AA)


def generer_video_debug(video_path, clip, chemin_clip, dossier_debug=None):
    """Génère la vidéo split-screen annotée pour un clip et renvoie son chemin."""
    dossier = dossier_debug or DOSSIER_DEBUG
    os.makedirs(dossier, exist_ok=True)
    racine = os.path.splitext(os.path.basename(chemin_clip))[0]
    chemin_debug = os.path.join(dossier, f"{racine}_debug.mp4")

    cap_src = cv2.VideoCapture(video_path)
    cap_out = cv2.VideoCapture(chemin_clip)
    if not cap_src.isOpened() or not cap_out.isOpened():
        cap_src.release()
        cap_out.release()
        raise RuntimeError(f"Impossible d'ouvrir les vidéos pour le rendu debug ({chemin_clip}).")

    fps = cap_out.get(cv2.CAP_PROP_FPS) or cap_src.get(cv2.CAP_PROP_FPS) or 30.0
    w_src = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_out = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Alignement sur la hauteur du clip final ; largeur totale paire (contrainte yuv420p)
    w_left = int(round(w_src * (h_out / max(1, h_src)))) // 2 * 2
    sep_w = 4
    total_w = (w_left + sep_w + w_out) // 2 * 2
    total_h = h_out // 2 * 2
    sep = np.full((total_h, total_w - w_left - w_out, 3), (35, 38, 46), dtype=np.uint8)

    rms_norm = _extraire_rms_clip(chemin_clip)
    duree = max(0.1, float(clip["fin"] - clip["debut"]))
    cap_src.set(cv2.CAP_PROP_POS_MSEC, clip["debut"] * 1000.0)

    cmd = [
        resoudre_ffmpeg(),
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{total_w}x{total_h}", "-r", f"{fps:.4f}", "-i", "-",
        "-i", chemin_clip,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest",
        chemin_debug, "-y", "-loglevel", "error",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    idx_frame = 0
    try:
        while True:
            ok_out, frame_out = cap_out.read()
            if not ok_out:
                break
            ok_src, frame_src = cap_src.read()
            if not ok_src:
                break
            t_rel = idx_frame / fps

            # Annoter dans le repère natif source puis redimensionner si besoin
            _dessiner_cadre_cadrage(frame_src, clip["cadrage"], clip.get("debug"), t_rel)
            _dessiner_boites(frame_src, clip.get("debug"), t_rel)
            _dessiner_hud_et_audio(frame_src, clip["cadrage"]["mode"], t_rel, duree, rms_norm)
            if frame_src.shape[1] != w_left or frame_src.shape[0] != total_h:
                frame_src = cv2.resize(frame_src, (w_left, total_h), interpolation=cv2.INTER_AREA)

            if frame_out.shape[0] != total_h:
                frame_out = cv2.resize(frame_out, (w_out, total_h), interpolation=cv2.INTER_AREA)
            _dessiner_badge_sortie(frame_out)

            composite = np.hstack([frame_src, sep, frame_out])
            proc.stdin.write(composite.tobytes())
            idx_frame += 1
    finally:
        cap_src.release()
        cap_out.release()
        if proc.stdin:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg debug a échoué (code {code}) : {stderr.strip()[-400:]}")

    print(f" -> Vidéo debug générée : {chemin_debug}")
    return chemin_debug


def generer_videos_debug(video_path, clips, chemins_clips, dossier_debug=None):
    """Génère les vidéos split-screen pour tous les clips d'une vidéo."""
    print(f"[{os.path.basename(video_path)}] Rendu des vidéos debug split-screen...")
    sorties = []
    for clip, chemin_clip in zip(clips, chemins_clips):
        sorties.append(generer_video_debug(video_path, clip, chemin_clip, dossier_debug=dossier_debug))
    return sorties
