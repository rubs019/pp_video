# Documentation technique — Machine à highlights

## 1. Vue d'ensemble

Pipeline Python organisé en **package `src/`** avec un point d'entrée `main.py` à la racine. Il télécharge des vidéos YouTube de combats et produit des highlights verticaux 9:16 enrichis de métadonnées et de légendes. Cible : **macOS Apple Silicon (M4)**, Python **3.14** (venv `.venv/`).

Chaîne de traitement en 4 étapes, orchestrée par vidéo, en « flux tendu » et **résiliente** (un échec isolé n'arrête pas le lot).

## 2. Structure du package

```
main.py                    Point d'entrée (argparse) → executer_pipeline_complet(url)
src/
├── config.py              Constantes ajustables + création des dossiers de travail
├── ffmpeg_utils.py        resoudre_ffmpeg(), lancer_ffmpeg()
├── telechargement.py      lister_videos_playlist(), telecharger_video()
├── audio.py               Étape 1 — analyser_audio_fast()
├── vision.py              Étape 2 — verifier_et_cadrer_locate_anything()
├── decoupage.py           Étape 3 — couper_et_recadrer_video()
├── enrichissement.py      Étape 4 — enrichir_metadonnees(), nommage, légendes, manifeste
└── pipeline.py            Orchestration — traiter_video(), executer_pipeline_complet()
```

**Graphe de dépendances** : les modules d'étape (`audio`, `vision`, `decoupage`, `telechargement`, `enrichissement`) ne dépendent que de `config` et `ffmpeg_utils`. `pipeline` dépend de tous. Aucun cycle. Imports relatifs (`from .config import ...`) → le code s'exécute comme package via `main.py`.

## 3. Flux de données

```
url ──▶ lister_videos_playlist ──▶ [urls]
                                     │  (pour chaque url)
                                     ▼
        telecharger_video ──▶ (chemin_mp4, info)        info = dict yt-dlp complet
                                     │
                                     ▼
        analyser_audio_fast ──▶ [(debut, fin), ...]      segments candidats (secondes)
                                     │
                                     ▼
   verifier_et_cadrer_locate_anything ──▶ [{debut, fin, crop_x}, ...]
                                     │
                                     ▼
        enrichir_metadonnees(info) ──▶ meta (dict)       1 appel LLM / vidéo
                                     │
              construire_base_nom(meta, id) ──▶ base_nom
                                     ▼
        couper_et_recadrer_video ──▶ [chemins_mp4]
                                     │
             ecrire_legende_txt (par clip) + maj_manifeste
```

## 4. Détail des étapes

### Étape 1 — Détection audio (`audio.py`)
1. Extraction WAV mono 22 kHz via ffmpeg.
2. Normalisation : conversion **float64 avant** mise au carré (évite l'overflow int16²), moyenne des canaux si stéréo, division par l'amplitude max (signal ramené à [-1, 1]).
3. Énergie **RMS par fenêtres de 0,5 s**, convertie en dB.
4. **Seuil adaptatif par percentile** : `seuil = max(percentile(rms_db, PERCENTILE_PIC), médiane + MARGE_MIN_DB)`. Choix clé — un seuil « médiane + N dB » fixe échoue sur l'audio de diffusion compressé (dynamique observée ~8 dB) ; le percentile s'adapte à la dynamique réelle, le plancher de marge évite les faux positifs sur un son uniforme.
5. Regroupement des fenêtres chaudes distantes de < `DUREE_HIGHLIGHT` en clusters ; on garde la fenêtre la plus intense comme pic.
6. Fenêtrage : **pré-roll 70 %** (14 s avant le pic, 6 s après, pour 20 s) — la clameur suit l'action, on capture donc ce qui précède le pic.
7. Fusion des segments qui se chevauchent. **Sortie : `[(debut, fin)]` en secondes.**

### Étape 2 — Validation visuelle (`vision.py`)
- Modèle `mlx-community/LocateAnything-3B-4bit` (VLM de grounding) via **mlx-vlm**, chargé **une seule fois** (cache module `_modele_vision`) et réutilisé sur toutes les vidéos ; `_liberer_memoire_mlx()` vide le cache MLX après inférence.
- Pour chaque segment : `FRAMES_PAR_SEGMENT` (3) frames extraites via **OpenCV** (pas ffmpeg), soumises au modèle avec le prompt `PROMPT_DETECTION`.
- Le modèle renvoie des boîtes au format `<ref>...</ref><box><x1><y1><x2><y2></box>` en coordonnées **0–1000**, parsées par regex.
- `_calculer_crop_x` : centre le crop 9:16 sur le barycentre horizontal des boîtes (converti en pixels), borné au cadre.
- Un segment **sans aucune détection est rejeté** (faux positif audio). **Sortie : `[{debut, fin, crop_x}]`.**

### Étape 3 — Découpe & recadrage (`decoupage.py`)
- Pour chaque clip : ffmpeg `-ss/-to` + filtre `crop=in_h*9/16:in_h:crop_x:0`, ré-encodage `libx264 -crf 22` + audio **`aac 128k`** (le ré-encodage audio est volontaire : l'audio YouTube est souvent en `opus`, mal supporté en conteneur MP4 par les plateformes).
- Nom de sortie = `{base_nom}_{i}.mp4`, protégé contre l'écrasement par `_chemin_unique` (suffixe `_2`, `_3`…). **Sortie : liste des chemins produits.**

### Étape 4 — Enrichissement (`enrichissement.py`)
- **1 seul appel LLM par vidéo** (les métadonnées sont au niveau vidéo, chaque clip en hérite).
- `_appeler_ollama` : `POST http://localhost:11434/api/generate` avec `format` = **schéma JSON** imposé, `stream=False`, `think=False` (désactive le raisonnement des modèles « thinking » ; repli automatique si le modèle refuse ce champ). Modèle par défaut `qwen3.5:9b`.
- `_extraire_json` : retire un éventuel bloc `<think>`, `json.loads`, repli sur extraction de la 1re accolade équilibrée.
- **Piège de la date neutralisé** : `_valider_date` n'accepte qu'une date `AAAA-MM-JJ` plausible issue du texte libre. `upload_date`/`timestamp` de yt-dlp (date de re-publication) ne sont **jamais** utilisés comme date d'événement.
- **Non bloquant** : `enrichir_metadonnees` ne lève jamais ; toute panne (Ollama down, JSON invalide) → repli `_enrichi=False` avec valeurs `inconnu`.
- **Sorties** :
  - `construire_base_nom` → `AAAA-MM-JJ_evenement_vainqueur-vs-perdant_methode` (slugifié, ASCII, minuscules ; repli sur l'id YouTube si non enrichi).
  - `ecrire_legende_txt` → un `.txt` par clip : 4 plateformes × {fr, en} + hashtags.
  - `maj_manifeste` → **`manifeste.json` = source de vérité** (upsert dédupliqué par `chemin`) ; **`manifeste.csv` régénéré intégralement** depuis le JSON à chaque écriture (cohérence + dédup garanties, même après plusieurs runs).

## 5. Orchestration & résilience (`pipeline.py`)

- `executer_pipeline_complet(url_source=None)` : `url_source` prime, sinon repli sur `config.URL_SOURCE`. Boucle sur `lister_videos_playlist`.
- Chaque vidéo est traitée dans un `try/except/finally` :
  - `except Exception` : log clair, on passe à la vidéo suivante (**résilience du lot**).
  - `finally` : suppression de la source (économie d'espace disque), même en cas d'erreur.
- `traiter_video` court-circuite proprement si l'étape 1 ou 2 ne produit rien.
- À l'opposé de cette tolérance : **`lancer_ffmpeg` échoue bruyamment** (`subprocess.run` + vérification du code retour + `RuntimeError` avec stderr) — remplace l'ancien `os.system` silencieux.

## 6. ffmpeg hors PATH

`resoudre_ffmpeg()` localise le binaire via `shutil.which`, puis des replis Homebrew (`/opt/homebrew/bin/ffmpeg`, `/usr/local/bin/ffmpeg`), et lève une erreur explicite sinon. Nécessaire car ffmpeg est souvent installé hors du PATH du venv/pyenv. `telecharger_video` passe aussi `ffmpeg_location` à yt-dlp (le merge vidéo+audio en a besoin).

## 7. Dépendances

Épinglées dans `requirements.txt` (versions testées) :
- `numpy`, `scipy` (audio), `yt-dlp` (téléchargement), `opencv-python` (frames), `requests` (Ollama).
- `mlx`, `mlx-vlm` avec **marqueur de plateforme** `sys_platform=="darwin" and platform_machine=="arm64"` (ignorés hors Apple Silicon).
- **Hors pip** : `ffmpeg` (Homebrew) et le serveur **Ollama** + un modèle instruct.
- Le modèle vision (~3,2 Go) est téléchargé au 1er run par mlx-vlm.

## 8. Décisions de conception (résumé)

| Décision | Raison |
|----------|--------|
| Percentile adaptatif (pas seuil dB fixe) | Robuste à la dynamique très variable des sources. |
| Métadonnées au niveau vidéo, 1 appel LLM | Vidéos mono-combat ; évite N appels redondants. |
| Enrichissement non bloquant | Une panne LLM ne doit pas faire perdre les clips. |
| Modèle vision chargé une fois | Coût de chargement (~3 Go) amorti sur tout le lot. |
| JSON source de vérité, CSV régénéré | Dédup et cohérence garanties entre les deux formats. |
| ffmpeg résolu explicitement | Indépendance vis-à-vis du PATH du shell. |
| Audio ré-encodé en AAC | Compatibilité plateformes (opus-en-MP4 mal supporté). |

## 9. Points d'extension

- **Changer de modèle de légendes** : `MODELE_LLM` dans `config.py` (tout modèle Ollama instruct).
- **Ajouter/retirer une plateforme ou une langue** : `PLATEFORMES` / `LANGUES_LEGENDES` (+ ajuster le schéma dans `_schema_enrichissement` si nouvelle plateforme).
- **Remplacer la validation visuelle** (contrainte de licence LocateAnything) : réécrire `vision.py` en conservant la signature `verifier_et_cadrer_locate_anything(video_path, candidats) -> [{debut, fin, crop_x}]`. Le reste du pipeline est inchangé.
- **Cibler un autre sport / type de contenu** : ajuster `PROMPT_DETECTION` (grounding) et le prompt d'enrichissement (`_construire_prompt_enrichissement`).

## 10. Vérification

Pas de suite de tests automatisés dans le repo. Vérifications manuelles disponibles :
```bash
python3 -m py_compile src/*.py main.py            # syntaxe de tous les modules
.venv/bin/python -c "from src.pipeline import executer_pipeline_complet"   # câblage des imports
.venv/bin/python main.py "https://www.youtube.com/watch?v=..."             # run réel sur une vidéo
```
Le pipeline a été validé de bout en bout sur un cas réel (UFC 239, Masvidal–Askren) : détection audio → 2 moments, validation visuelle → 2/2, clips 404×720 rendus, légendes bilingues et manifeste générés.
