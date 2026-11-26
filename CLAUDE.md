# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vue d'ensemble

Pipeline (package `src/`, point d'entrée `main.py` à la racine) qui, à partir d'une playlist YouTube de combats (UFC/MMA/boxe), génère automatiquement des **highlights verticaux 9:16** prêts pour les réseaux sociaux, chacun accompagné de métadonnées et de légendes bilingues. Conçu pour tourner sur **Mac Apple Silicon (M4)** en « flux tendu » (une vidéo téléchargée → traitée → source supprimée → suivante).

## Structure des modules (`src/`)

| Module | Rôle |
|--------|------|
| `config.py` | **Toutes** les constantes ajustables (URL, dossiers, seuils, modèles, plateformes). Crée les dossiers de travail à l'import. |
| `ffmpeg_utils.py` | `resoudre_ffmpeg` (localise le binaire hors PATH), `lancer_ffmpeg` (exécution sans shell, échec bruyant). |
| `telechargement.py` | `lister_videos_playlist`, `telecharger_video` (yt-dlp). |
| `audio.py` | Étape 1 — `analyser_audio_fast`. |
| `vision.py` | Étape 2 — `verifier_et_cadrer_vision` (+ chargement modèle MLX Qwen3-VL). |
| `decoupage.py` | Étape 3 — `couper_et_recadrer_video`. |
| `debug_visuel.py` | Option `--debug-video` — `generer_videos_debug` (split-screen 16:9 annoté \| 9:16 final dans `highlights_prets/debug/`). |
| `enrichissement.py` | Étape 4 — `enrichir_metadonnees`, nommage enrichi, légendes `.txt`, manifeste. |
| `historique.py` | Registre des vidéos déjà traitées (`videos_traitees.json`) : `charger_historique`, `est_deja_traitee`, `marquer_traitee`. |
| `pipeline.py` | Orchestration — `traiter_video`, `executer_pipeline_complet`. |

Les modules d'étape ne dépendent que de `config` et `ffmpeg_utils` ; `pipeline` dépend de tous. Imports relatifs (`from .config import ...`) → à lancer comme package via `main.py`.

## Prérequis système (indispensables, sinon échec au runtime)

- **Python 3.14** dans `.venv/` (venv du projet ; utiliser `.venv/bin/python`).
- **ffmpeg** : requis par les étapes 1, 3 **et** le merge yt-dlp. Souvent installé via Homebrew (`/opt/homebrew/bin/ffmpeg`) **hors du PATH du shell** → `src/ffmpeg_utils.resoudre_ffmpeg` le localise explicitement, ne pas supposer qu'il est dans le PATH.
- **mlx-vlm** + modèle `mlx-community/Qwen3-VL-8B-Instruct-4bit` (~5,8 Go, Apache 2.0, téléchargé au 1er run, **Apple Silicon uniquement** ; variante `4B-Instruct-4bit` ~3,1 Go pour Mac 16 Go). Le stock `mlx-vlm` 0.6.3 suffit.
- **Ollama** : serveur local actif sur `http://localhost:11434` + un modèle instruct (défaut `mistral:7b`) pour l'étape 4.
- `requirements.txt` liste les dépendances pip épinglées aux versions testées ; `mlx`/`mlx-vlm` portent un marqueur `darwin`/`arm64` (ignorés hors Apple Silicon). ffmpeg reste hors pip (`brew install ffmpeg`).

## Commandes

```bash
# Vérifier la syntaxe de tous les modules (pas de suite de tests dans le repo)
python3 -m py_compile src/*.py main.py

# Lancer sur une playlist ou une vidéo — URL passée en argument
.venv/bin/python main.py "https://www.youtube.com/playlist?list=..."
.venv/bin/python main.py "https://www.youtube.com/watch?v=..."   # vidéo isolée (repli lister_videos_playlist)

# Sans argument : utilise URL_SOURCE de src/config.py (erreur claire si c'est encore le gabarit)
.venv/bin/python main.py

# Retraiter même les vidéos déjà enregistrées au registre
.venv/bin/python main.py "https://www.youtube.com/playlist?list=..." --force

# Générer aussi les vidéos démo split-screen (16:9 annoté | 9:16 final)
.venv/bin/python main.py "https://www.youtube.com/watch?v=..." --debug-video

# Interface web locale (Streamlit) : onglets Lancer / Galerie / Manifeste
.venv/bin/streamlit run interface_web.py
```

Pour un test isolé d'une étape sans re-télécharger, importer le module ciblé et appeler sa fonction (`from src.audio import analyser_audio_fast`, `from src.enrichissement import enrichir_metadonnees`, ...) sur un fichier de `combats_bruts/`. Les fonctions sont découplées et réutilisables ainsi.

## Architecture (pipeline en 4 étapes)

`executer_pipeline_complet()` (`src/pipeline.py`, appelé par `main.py`) boucle sur `lister_videos_playlist(URL_SOURCE)` puis, par vidéo : `telecharger_video` → `traiter_video`, qui enchaîne :

1. **`analyser_audio_fast`** (`audio.py`) — extrait le WAV (ffmpeg), calcule la RMS par fenêtres de 0,5 s, détecte les pics par **seuil de percentile auto-adaptatif** (`PERCENTILE_PIC`), regroupe en segments centrés sur le pic avec pré-roll 70 %. Retourne des `(debut, fin)`.
2. **`verifier_et_cadrer_vision`** (`vision.py`) — échantillonne des frames (OpenCV, redimensionnées à `VISION_LARGEUR_MAX=640`), les passe à Qwen3-VL (MLX, Apache 2.0) avec un **prompt ciblé** (`PROMPT_DETECTION`) qui **isole les combattants en action** (écarte arbitre/coins/public et rejette intro/interviews/décision). Calcule un **cadrage adaptatif** (`_calculer_cadrage`) : crop 9:16 serré centré sur les combattants, ou zoom arrière + fond flou s'ils sont écartés. Rejette les faux positifs audio. Retourne des `{debut, fin, cadrage}`. Optionnel — `RECADRAGE_DYNAMIQUE=True` : **panoramique fluide** (le crop 9:16 suit les combattants dans le temps via une expression ffmpeg ; repli statique s'ils sont trop écartés).
3. **`couper_et_recadrer_video`** (`decoupage.py`) — découpe et recadre en 9:16 via ffmpeg, avec un nom de fichier enrichi (voir étape 4). Retourne les chemins produits.
4. **`enrichir_metadonnees`** (`enrichissement.py`) — 1 appel LLM Ollama qui extrait les métadonnées (combattants, événement, date, poids, méthode) et rédige des légendes bilingues FR/EN pour 4 plateformes. Produit : noms de fichiers enrichis, un `.txt` de légendes **par vidéo** (`{base_nom}.txt`, commun à ses clips), et un manifeste global.

## Décisions de conception à connaître (non évidentes)

- **Résilience par vidéo** : la boucle de `executer_pipeline_complet` enveloppe chaque vidéo dans `try/except/finally` — un échec isolé n'arrête pas le lot, et la source est toujours supprimée. À l'inverse, `lancer_ffmpeg` **échoue bruyamment** (pas d'échec silencieux `os.system`).
- **L'enrichissement (étape 4) ne doit jamais bloquer** : `enrichir_metadonnees` ne lève jamais ; si Ollama est indisponible ou renvoie un JSON invalide, elle renvoie un repli `_enrichi=False` et les clips sont quand même produits (nom basé sur l'id YouTube).
- **Détection audio par percentile, pas par seuil dB fixe** : un « médiane + N dB » casse sur l'audio de diffusion compressé (dynamique ~8 dB). Ne pas régresser vers un seuil absolu.
- **Recadrage dynamique (option `RECADRAGE_DYNAMIQUE`, défaut False)** : si activé, `vision.py` détecte les combattants toutes les `INTERVALLE_SUIVI` s, lisse la trajectoire (moyenne glissante + limite de vitesse) et produit `{"mode":"pan","expr":…}` ; `decoupage.py` rend un crop dont l'offset `x` est une **expression de `t`** (crop qui glisse). Ce mode exige un **seek en entrée** (`-ss debut -i video`) + `setpts=PTS-STARTPTS` pour que `t` parte de 0 (les modes crop/pad gardent le trim en sortie). Coût ~10 détections/clip ; repli sur le cadrage statique si les combattants sont trop écartés (`fit < SEUIL_PAN_FIT`).
- **Piège de la date** : `upload_date`/`timestamp` de yt-dlp = date de re-publication, **PAS** la date du combat. La vraie date n'est extraite que du texte libre (titre/description) et validée par `_valider_date` (format `AAAA-MM-JJ`).
- **Métadonnées au niveau vidéo** : les vidéos sont mono-combat → un seul appel LLM par vidéo, tous les clips héritent des mêmes métadonnées (+ leur index). Ne pas appeler le LLM par clip.
- **Modèle vision chargé une seule fois** (`_charger_modele_vision`, cache global `_modele_vision` dans `vision.py`) et réutilisé sur toutes les vidéos ; `_liberer_memoire_mlx` vide le cache MLX après inférence.
- **Manifeste** : `manifeste.json` est la **source de vérité** (upsert dédupliqué par `chemin`) ; `manifeste.csv` est **régénéré intégralement** depuis le JSON à chaque écriture.
- **Registre des vidéos traitées** (`src/historique.py`, fichier `highlights_prets/videos_traitees.json`, indexé par ID YouTube) : `executer_pipeline_complet` **saute avant téléchargement** toute vidéo déjà enregistrée. Une vidéo est marquée dès qu'elle est analysée jusqu'au bout (même 0 clip), **jamais** sur échec de téléchargement ou plantage (→ retry). `--force` ignore le registre. Au 1er usage, le registre s'amorce depuis les `video_id` de `manifeste.json`.
- Toute la configuration se modifie dans **`src/config.py`** — c'est le seul endroit à toucher pour ajuster le comportement.

## Licence

Le modèle **Qwen3-VL-8B-Instruct-4bit est sous licence Apache 2.0** (usage commercial autorisé). Pour tout usage commercial, seuls les droits sur les vidéos sources restent à la charge de l'utilisateur / client.

## Point technique
- Ne te mets jamais en co-auteur lors de la création d'un commit
