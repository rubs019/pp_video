"""Étape 4 : enrichissement métadonnées + légendes bilingues via un LLM local Ollama.

Produit aussi les noms de fichiers enrichis, les .txt de légendes par clip et le manifeste global.
Ne lève jamais : en cas de panne LLM, on retombe sur un repli 'inconnu' non bloquant.
"""
import csv
import json
import os
import re
import unicodedata

import requests

from .config import (LANGUES_LEGENDES, MANIFESTE_CSV, MANIFESTE_JSON, MODELE_LLM,
                     NOM_FICHIER_MAX, OLLAMA_NUM_CTX, OLLAMA_TIMEOUT, OLLAMA_URL,
                     PLATEFORMES, VALEUR_INCONNUE)


def _appeler_ollama(prompt, schema):
    """POST bas niveau vers Ollama. Renvoie la chaîne 'response'. Gère les modèles sans 'think'."""
    charge = {
        "model": MODELE_LLM,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX},
    }
    reponse = requests.post(OLLAMA_URL, json=charge, timeout=OLLAMA_TIMEOUT)
    if reponse.status_code == 400 and "think" in reponse.text.lower():
        charge.pop("think")  # modèle sans mode raisonnement : on réémet sans ce champ
        reponse = requests.post(OLLAMA_URL, json=charge, timeout=OLLAMA_TIMEOUT)
    reponse.raise_for_status()
    return reponse.json().get("response", "")


def _extraire_json(texte):
    """Extrait un objet JSON d'une sortie LLM (retire un bloc <think>, tolère du texte autour)."""
    texte = re.sub(r"<think>.*?</think>", "", texte, flags=re.DOTALL).strip()
    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        debut, fin = texte.find("{"), texte.rfind("}")
        if debut != -1 and fin > debut:
            return json.loads(texte[debut:fin + 1])
        raise ValueError("Aucun JSON exploitable dans la réponse du LLM.")


def _valider_date(valeur):
    """N'accepte qu'une date AAAA-MM-JJ plausible ; sinon None (jamais l'upload_date)."""
    if not isinstance(valeur, str):
        return None
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", valeur.strip())
    if not m:
        return None
    annee, mois, jour = map(int, m.groups())
    if 1990 <= annee <= 2100 and 1 <= mois <= 12 and 1 <= jour <= 31:
        return valeur.strip()
    return None


def _schema_enrichissement():
    """Schéma JSON imposé à Ollama pour une sortie structurée fiable."""
    legende_std = {
        "type": "object",
        "properties": {"texte": {"type": "string"},
                       "hashtags": {"type": "array", "items": {"type": "string"}}},
        "required": ["texte", "hashtags"],
    }
    legende_yt = {
        "type": "object",
        "properties": {"titre": {"type": "string"}, "description": {"type": "string"},
                       "hashtags": {"type": "array", "items": {"type": "string"}}},
        "required": ["titre", "description", "hashtags"],
    }

    def bilingue(modele):
        return {"type": "object", "properties": {"fr": modele, "en": modele},
                "required": ["fr", "en"]}

    return {
        "type": "object",
        "properties": {
            "combattants": {"type": "array", "items": {"type": "string"}},
            "vainqueur": {"type": "string"},
            "perdant": {"type": "string"},
            "evenement": {"type": "string"},
            "date_combat": {"type": "string"},
            "categorie_poids": {"type": "string"},
            "methode": {"type": "string"},
            "resume": {"type": "string"},
            "legendes": {
                "type": "object",
                "properties": {
                    "instagram_reels": bilingue(legende_std),
                    "tiktok": bilingue(legende_std),
                    "youtube_shorts": bilingue(legende_yt),
                    "x": bilingue(legende_std),
                },
                "required": PLATEFORMES,
            },
        },
        "required": ["combattants", "vainqueur", "perdant", "evenement",
                     "date_combat", "categorie_poids", "methode", "legendes"],
    }


def _construire_prompt_enrichissement(info):
    """Construit le prompt d'extraction à partir des métadonnées YouTube (texte libre)."""
    titre = info.get("title", "")
    description = (info.get("description") or "").strip()
    # On ne garde que le 1er paragraphe éditorial (le reste = liens/boilerplate promo).
    description = re.split(r"\n\s*\n|Subscribe", description, maxsplit=1)[0].strip()
    tags = ", ".join((info.get("tags") or [])[:40])
    plateformes = ", ".join(PLATEFORMES)
    return (
        "Tu prépares des publications pour des clips de sports de combat (UFC, MMA, boxe). "
        "À partir des métadonnées YouTube ci-dessous, renvoie UNIQUEMENT un JSON conforme au schéma.\n\n"
        f"TITRE : {titre}\n"
        f"DESCRIPTION : {description}\n"
        f"TAGS : {tags}\n\n"
        "Consignes :\n"
        "- Identifie combattants, vainqueur, perdant, événement, catégorie de poids et méthode "
        "de victoire (KO, soumission, décision...).\n"
        "- date_combat : la date RÉELLE du combat au format AAAA-MM-JJ si déductible du texte ; "
        "sinon chaîne vide. N'invente pas et n'utilise jamais une date de mise en ligne.\n"
        f"- Pour CHAQUE plateforme ({plateformes}), rédige une légende accrocheuse en FRANÇAIS (fr) "
        "ET en ANGLAIS (en), avec des hashtags adaptés. YouTube Shorts = titre + description ; "
        "les autres = texte court.\n"
        "- Si une information est absente, laisse une chaîne vide plutôt que d'inventer."
    )


def _repli_metadonnees():
    """Métadonnées 'inconnu' quand l'enrichissement échoue : jamais bloquant."""
    return {
        "_enrichi": False, "combattants": [],
        "vainqueur": VALEUR_INCONNUE, "perdant": VALEUR_INCONNUE,
        "evenement": VALEUR_INCONNUE, "date_combat": None,
        "categorie_poids": VALEUR_INCONNUE, "methode": VALEUR_INCONNUE,
        "resume": "", "legendes": {},
    }


def enrichir_metadonnees(info):
    """Étape 4 : métadonnées + légendes via Ollama. NE LÈVE JAMAIS (repli propre sur panne)."""
    print("    Étape 4 : Enrichissement métadonnées (Ollama)...")
    try:
        brut = _appeler_ollama(_construire_prompt_enrichissement(info), _schema_enrichissement())
        meta = _extraire_json(brut)
    except Exception as e:
        print(f"    -> Enrichissement indisponible ({e}). Repli métadonnées 'inconnu'.")
        return _repli_metadonnees()
    meta["_enrichi"] = True
    meta["date_combat"] = _valider_date(meta.get("date_combat"))
    meta.setdefault("combattants", [])
    meta.setdefault("legendes", {})
    print(f"    -> {meta.get('vainqueur', '?')} vs {meta.get('perdant', '?')} "
          f"@ {meta.get('evenement', '?')} ({meta.get('date_combat') or 'date inconnue'})")
    return meta


def _slugifier(txt, longueur_max=NOM_FICHIER_MAX):
    """Translittère en ASCII, minuscules, tirets ; borne la longueur."""
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s-]", "", txt).strip().lower()
    txt = re.sub(r"[\s_-]+", "-", txt)
    return txt[:longueur_max].strip("-")


def _nom_de_famille(nom):
    """Dernier mot d'un nom complet (ex. 'Jorge Masvidal' -> 'Masvidal')."""
    nom = (nom or "").strip()
    return nom.split()[-1] if nom else ""


def construire_base_nom(meta, youtube_id):
    """Nom de base enrichi : AAAA-MM-JJ_evenement_vainqueur-vs-perdant_methode. Repli : id YouTube."""
    if not meta.get("_enrichi"):
        return youtube_id or "highlight"
    morceaux = []
    if meta.get("date_combat"):
        morceaux.append(meta["date_combat"])
    if meta.get("evenement") and meta["evenement"] != VALEUR_INCONNUE:
        morceaux.append(_slugifier(meta["evenement"], 20))
    vainqueur, perdant = _nom_de_famille(meta.get("vainqueur")), _nom_de_famille(meta.get("perdant"))
    if vainqueur and perdant:
        morceaux.append(f"{_slugifier(vainqueur, 20)}-vs-{_slugifier(perdant, 20)}")
    if meta.get("methode") and meta["methode"] != VALEUR_INCONNUE:
        morceaux.append(_slugifier(meta["methode"], 15))
    return "_".join(m for m in morceaux if m) or youtube_id or "highlight"


def ecrire_legende_txt(chemin_clip, meta, index):
    """Écrit un .txt prêt à publier (4 plateformes × FR/EN) à côté du clip."""
    chemin_txt = os.path.splitext(chemin_clip)[0] + ".txt"
    legendes = meta.get("legendes") or {}
    entete = (f"# {meta.get('vainqueur', VALEUR_INCONNUE)} vs {meta.get('perdant', VALEUR_INCONNUE)}"
              f" — {meta.get('evenement', VALEUR_INCONNUE)} "
              f"({meta.get('date_combat') or 'date inconnue'})")
    lignes = [entete, f"# Clip {index} — {os.path.basename(chemin_clip)}", ""]

    if not meta.get("_enrichi") or not legendes:
        lignes.append("(Enrichissement indisponible — légendes non générées.)")
    else:
        titres = {"instagram_reels": "INSTAGRAM REELS", "tiktok": "TIKTOK",
                  "youtube_shorts": "YOUTUBE SHORTS", "x": "X (TWITTER)"}
        for plateforme in PLATEFORMES:
            bloc = legendes.get(plateforme) or {}
            lignes.append(f"===== {titres.get(plateforme, plateforme.upper())} =====")
            for langue in LANGUES_LEGENDES:
                contenu = bloc.get(langue) or {}
                lignes.append(f"[{langue.upper()}]")
                if "titre" in contenu:  # YouTube Shorts
                    lignes.append(f"Titre : {contenu.get('titre', '')}")
                    lignes.append(f"Description : {contenu.get('description', '')}")
                else:
                    lignes.append(contenu.get("texte", ""))
                hashtags = contenu.get("hashtags") or []
                if hashtags:
                    lignes.append(" ".join(h if h.startswith("#") else f"#{h}" for h in hashtags))
                lignes.append("")
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes))
    return chemin_txt


def _lignes_manifeste(clips_chemins, meta, info):
    """Une ligne de manifeste par clip : métadonnées communes + chemins propres au clip."""
    lignes = []
    for i, chemin in enumerate(clips_chemins):
        lignes.append({
            "chemin": chemin,
            "fichier": os.path.basename(chemin),
            "legende_txt": os.path.splitext(chemin)[0] + ".txt",
            "video_id": info.get("id", ""),
            "url_source": info.get("webpage_url", ""),
            "combattants": ", ".join(meta.get("combattants") or []),
            "vainqueur": meta.get("vainqueur", VALEUR_INCONNUE),
            "perdant": meta.get("perdant", VALEUR_INCONNUE),
            "evenement": meta.get("evenement", VALEUR_INCONNUE),
            "date_combat": meta.get("date_combat") or "",
            "categorie_poids": meta.get("categorie_poids", VALEUR_INCONNUE),
            "methode": meta.get("methode", VALEUR_INCONNUE),
            "resume": meta.get("resume", ""),
            "enrichi": meta.get("_enrichi", False),
            "index_clip": i,
        })
    return lignes


def maj_manifeste(clips_chemins, meta, info):
    """JSON = source de vérité (upsert dédupliqué par 'chemin') ; CSV régénéré intégralement."""
    nouvelles_lignes = _lignes_manifeste(clips_chemins, meta, info)
    lignes = []
    if os.path.exists(MANIFESTE_JSON):
        try:
            with open(MANIFESTE_JSON, encoding="utf-8") as f:
                lignes = json.load(f)
        except (json.JSONDecodeError, OSError):
            lignes = []
    index = {l["chemin"]: l for l in lignes if isinstance(l, dict) and "chemin" in l}
    for ligne in nouvelles_lignes:
        index[ligne["chemin"]] = ligne  # upsert : pas de doublon par chemin
    lignes = list(index.values())
    with open(MANIFESTE_JSON, "w", encoding="utf-8") as f:
        json.dump(lignes, f, ensure_ascii=False, indent=2)
    if lignes:
        colonnes = []
        for ligne in lignes:
            for cle in ligne:
                if cle not in colonnes:
                    colonnes.append(cle)
        with open(MANIFESTE_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(lignes)
    return len(lignes)
