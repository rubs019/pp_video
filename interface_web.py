"""Interface web locale (Streamlit) pour la machine à highlights.

Lancer :
    .venv/bin/streamlit run interface_web.py

Trois onglets : Lancer (soumettre une URL + logs en direct), Galerie (clips + légendes),
Manifeste & registre (tableaux). Le pipeline est exécuté via `main.py` en sous-processus,
de sorte que les modèles lourds (MLX, Ollama) restent hors du process Streamlit.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))
from src.config import (DOSSIER_SORTIE, FICHIER_HISTORIQUE, MANIFESTE_CSV,  # noqa: E402
                        MANIFESTE_JSON, URL_SOURCE)

st.set_page_config(page_title="Machine à highlights", page_icon="🎬", layout="wide")
st.title("🎬 Machine à highlights")


def _lire_json(chemin, defaut):
    p = Path(chemin)
    if not p.exists():
        return defaut
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return defaut


# --- Barre latérale : état rapide ---
clips_existants = sorted(Path(DOSSIER_SORTIE).glob("*.mp4"))
registre = _lire_json(FICHIER_HISTORIQUE, {})
_manifeste = _lire_json(MANIFESTE_JSON, [])
# Légende par clip : désormais une légende PAR VIDÉO (partagée par ses clips), via le manifeste.
legende_par_fichier = {l.get("fichier"): l.get("legende_txt")
                       for l in _manifeste if isinstance(l, dict)}
with st.sidebar:
    st.header("État")
    st.metric("Clips générés", len(clips_existants))
    st.metric("Vidéos traitées", len(registre))
    st.caption(f"Sorties : `{DOSSIER_SORTIE}`")
    if URL_SOURCE and "ID_DE_LA_PLAYLIST" not in URL_SOURCE:
        st.caption(f"URL par défaut : {URL_SOURCE}")

onglet_run, onglet_galerie, onglet_manifeste = st.tabs(
    ["▶️ Lancer", "🎞️ Galerie", "📋 Manifeste & registre"])

# --- Onglet 1 : lancer un traitement ---
with onglet_run:
    st.subheader("Lancer un traitement")
    url = st.text_input(
        "URL de la playlist ou de la vidéo YouTube",
        placeholder="https://www.youtube.com/playlist?list=…  ou  …/watch?v=…",
    )
    force = st.checkbox("Forcer le retraitement des vidéos déjà traitées (--force)")

    if st.button("Lancer", type="primary", disabled=not url.strip()):
        commande = [sys.executable, "main.py", url.strip()]
        if force:
            commande.append("--force")
        st.caption("Commande : `" + " ".join(commande) + "`")
        with st.status("Traitement en cours…", expanded=True) as statut:
            zone = st.empty()
            lignes = []
            proc = subprocess.Popen(
                commande, cwd=str(RACINE),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for ligne in proc.stdout:
                lignes.append(ligne.rstrip("\n"))
                zone.code("\n".join(lignes[-500:]))  # logs en direct (500 dernières lignes)
            code = proc.wait()
            if code == 0:
                statut.update(label="Traitement terminé ✅", state="complete")
            else:
                statut.update(label=f"Échec (code {code})", state="error")
        st.info("Ouvrez l'onglet **Galerie** pour voir les nouveaux clips.")

# --- Onglet 2 : galerie de clips + légendes ---
with onglet_galerie:
    st.subheader(f"Clips générés ({len(clips_existants)})")
    if not clips_existants:
        st.info("Aucun clip pour l'instant. Lancez un traitement dans l'onglet « Lancer ».")
    for clip in clips_existants:
        st.markdown(f"**{clip.name}**")
        col_video, col_legende = st.columns([1, 2])
        with col_video:
            st.video(str(clip))
            st.download_button("⬇️ Télécharger le clip", clip.read_bytes(),
                               file_name=clip.name, mime="video/mp4",
                               key=f"dl_{clip.name}")
        with col_legende:
            # Légende de la vidéo (via manifeste) ; repli : nom de base sans le suffixe _N.
            nom_txt = legende_par_fichier.get(clip.name)
            txt = Path(nom_txt) if nom_txt else clip.with_name(re.sub(r"_\d+$", "", clip.stem) + ".txt")
            if txt.exists():
                st.code(txt.read_text(encoding="utf-8"), language="markdown")
            else:
                st.caption("Pas de légende associée.")
        st.divider()

# --- Onglet 3 : manifeste + registre ---
with onglet_manifeste:
    st.subheader("Manifeste des clips")
    manifeste = _lire_json(MANIFESTE_JSON, [])
    if manifeste:
        st.dataframe(manifeste, use_container_width=True, hide_index=True)
        csv = Path(MANIFESTE_CSV)
        if csv.exists():
            st.download_button("⬇️ Télécharger le manifeste (CSV)", csv.read_bytes(),
                               file_name="manifeste.csv", mime="text/csv")
    else:
        st.info("Manifeste vide.")

    st.subheader("Vidéos déjà traitées (registre)")
    if registre:
        lignes = [{"video_id": vid, **infos} for vid, infos in registre.items()]
        st.dataframe(lignes, use_container_width=True, hide_index=True)
        st.caption("Ces vidéos sont ignorées automatiquement. "
                   "Cochez « Forcer » dans l'onglet « Lancer » pour les retraiter.")
    else:
        st.info("Registre vide.")
