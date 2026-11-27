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
├── vision.py              Étape 2 — verifier_et_cadrer_vision()
├── decoupage.py           Étape 3 — couper_et_recadrer_video()
├── debug_visuel.py        Mode --debug-video — rendu split-screen (16:9 annoté | 9:16 final)
├── enrichissement.py      Étape 4 — enrichir_metadonnees(), nommage, légendes, manifeste
├── historique.py          Registre des vidéos déjà traitées (videos_traitees.json)
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
   verifier_et_cadrer_vision ──▶ [{debut, fin, cadrage}, ...]
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
- Modèle `mlx-community/Qwen3-VL-8B-Instruct-4bit` (VLM Apache 2.0, usage commercial autorisé) via **mlx-vlm**, chargé **une seule fois** (cache module `_modele_vision`) et réutilisé sur toutes les vidéos ; `_liberer_memoire_mlx()` vide le cache MLX après inférence. Sur Mac 16 Go, basculer `MODELE_VISION` sur `mlx-community/Qwen3-VL-4B-Instruct-4bit`.
- Pour chaque segment : les frames extraites via **OpenCV** sont redimensionnées à `VISION_LARGEUR_MAX` (`640` px) avant envoi au VLM (les coordonnées `0–1000` étant relatives, la résolution réelle de la vidéo est conservée pour FFmpeg), puis nettoyées dans un bloc `try/finally`.
- Le modèle est guidé par `PROMPT_DETECTION` pour localiser uniquement les **combattants en action** (exclusion arbitre/coachs/public et rejet des scènes hors combat : intro, walkout, interview, décision) et renvoie un JSON compact `[{"bbox_2d": [x1, y1, x2, y2], "label": "f"}]` en coordonnées **0–1000**, parsé par `_parser_boites` (JSON strict avec repli regex).
- `_filtrer_combattants` garde les ≤2 plus grandes boîtes dans la bande utile (5–95 %) ; `_calculer_cadrage` en déduit un **cadrage adaptatif** : crop 9:16 serré centré sur leur milieu s'ils tiennent (ou s'ils sont très écartés au-delà d'un plafond → crop recentré lisible plutôt qu'un dézoom illisible), sinon zoom arrière + fond flou pour montrer les deux.
- Un segment **sans aucune détection est rejeté** (faux positif audio ou scène hors combat). **Sortie : `[{debut, fin, cadrage}]`.**
- **Recadrage dynamique** (option `RECADRAGE_DYNAMIQUE`) : au lieu d'un cadrage figé, on échantillonne toutes les `INTERVALLE_SUIVI` s, on suit le centre des combattants, on **lisse** la trajectoire (`_lisser_trajectoire` : moyenne glissante + limite de vitesse anti à-coups) et on génère une **expression `x(t)`** (`_construire_expr_pan`, interpolation linéaire par morceaux + offset pair) → cadrage `{"mode":"pan","expr":…}`. Repli sur le cadrage statique si les combattants sont trop écartés (`fit < SEUIL_PAN_FIT`).

### Étape 3 — Découpe & recadrage (`decoupage.py`)
- Pour chaque clip : ffmpeg + filtre issu du cadrage (`_filtre_cadrage` : crop 9:16 serré ; crop élargi + fond flou via `split`/`gblur`/`overlay` ; ou **panoramique** `setpts=PTS-STARTPTS,crop=w:h:'x(t)':0`), ré-encodage `libx264 -crf 22` + audio **`aac 128k`** (ré-encodage audio volontaire : l'audio YouTube est souvent en `opus`, mal supporté en MP4).
- **Seek selon le mode** : `pan` utilise un **seek en entrée** (`-ss debut -i video -t duree`) pour que `t` parte de 0 (indispensable à l'expression `x(t)`) ; `crop`/`pad` gardent le **trim en sortie** (`-i video -ss debut -to fin`), frame-exact.
- Nom de sortie = `{base_nom}_{i}.mp4`, protégé contre l'écrasement par `_chemin_unique` (suffixe `_2`, `_3`…). **Sortie : liste des chemins produits.**
- **Rendu debug split-screen (`debug_visuel.py`, option `--debug-video` / `DEBUG_VIDEO`)** : sans surcoût d'inférence, `vision.py` attache les métadonnées de suivi (`t_list`, boîtes brutes/filtrées, trajectoire `x_list`) à chaque clip. Si activé, `generer_videos_debug` assemble en streaming via pipe FFmpeg la vidéo source annotée à gauche (boîtes VLM, zone hors cadrage assombrie, rectangle 9:16 `pan`/`crop`/`pad`, bandeau HUD, courbe audio RMS temps réel) et le clip final 9:16 à droite, avec copie directe de la piste audio AAC, dans `highlights_prets/debug/{base_nom}_{i}_debug.mp4`.

### Étape 4 — Enrichissement (`enrichissement.py`)
- **1 seul appel LLM par vidéo** (les métadonnées sont au niveau vidéo, chaque clip en hérite).
- `_appeler_ollama` : `POST http://localhost:11434/api/generate` avec `format` = **schéma JSON** imposé, `stream=False`, `think=False` (désactive le raisonnement des modèles « thinking » ; repli automatique si le modèle refuse ce champ). Modèle par défaut `mistral:7b`.
- `_extraire_json` : retire un éventuel bloc `<think>`, `json.loads`, repli sur extraction de la 1re accolade équilibrée.
- **Piège de la date neutralisé** : `_valider_date` n'accepte qu'une date `AAAA-MM-JJ` plausible issue du texte libre. `upload_date`/`timestamp` de yt-dlp (date de re-publication) ne sont **jamais** utilisés comme date d'événement.
- **Non bloquant** : `enrichir_metadonnees` ne lève jamais ; toute panne (Ollama down, JSON invalide) → repli `_enrichi=False` avec valeurs `inconnu`.
- **Sorties** :
  - `construire_base_nom` → `AAAA-MM-JJ_evenement_vainqueur-vs-perdant_methode` (slugifié, ASCII, minuscules ; repli sur l'id YouTube si non enrichi).
  - `ecrire_legende_video` → un `.txt` **par vidéo** (`{base_nom}.txt`, en-tête listant les clips + 4 plateformes × {fr, en} + hashtags). Les légendes étant au niveau vidéo, un seul fichier au lieu d'un doublon par clip.
  - `maj_manifeste` → **`manifeste.json` = source de vérité** (upsert dédupliqué par `chemin`) ; **`manifeste.csv` régénéré intégralement** depuis le JSON à chaque écriture (cohérence + dédup garanties, même après plusieurs runs).

## 5. Orchestration & résilience (`pipeline.py`)

- `executer_pipeline_complet(url_source=None, force=False)` : `url_source` prime, sinon repli sur `config.URL_SOURCE`. Boucle sur `lister_videos_playlist` (qui renvoie des paires `(video_id, url)`).
- **Registre des vidéos traitées** (`historique.py`, fichier `videos_traitees.json` indexé par ID) : avant téléchargement, une vidéo déjà enregistrée est **ignorée** (`est_deja_traitee`) sauf si `force=True`. Après un `traiter_video` mené à terme (même 0 clip), la vidéo est enregistrée (`marquer_traitee` : date + titre + nb_clips) ; **jamais** sur échec de téléchargement ou plantage (→ retry au prochain run). Amorçage la 1re fois depuis les `video_id` de `manifeste.json`.
- Chaque vidéo est traitée dans un `try/except/finally` :
  - `except Exception` : log clair, on passe à la vidéo suivante (**résilience du lot**).
  - `finally` : suppression de la source (économie d'espace disque), même en cas d'erreur.
- `traiter_video` court-circuite proprement si l'étape 1 ou 2 ne produit rien.
- À l'opposé de cette tolérance : **`lancer_ffmpeg` échoue bruyamment** (`subprocess.run` + vérification du code retour + `RuntimeError` avec stderr) — remplace l'ancien `os.system` silencieux.

## 6. Binaires hors PATH & téléchargement YouTube (`ffmpeg`, `deno`, `cookies.txt`)

- **`ffmpeg`** : `resoudre_ffmpeg()` localise le binaire via `shutil.which`, puis des replis Homebrew (`/opt/homebrew/bin/ffmpeg`, `/usr/local/bin/ffmpeg`), et lève une erreur explicite sinon. Nécessaire car ffmpeg est souvent installé hors du PATH du venv/pyenv. `telecharger_video` passe aussi `ffmpeg_location` à yt-dlp (le merge vidéo+audio en a besoin).
- **Runtime JS (`deno` / `node`) pour `yt-dlp-ejs`** : YouTube exige un runtime JavaScript supporté pour résoudre les challenges de formats (`n challenge`). `_options_communes()` dans `src/telechargement.py` détecte automatiquement `deno` (recommandé, `brew install deno`) et `node` via `_trouver_binaire()` (PATH puis `/opt/homebrew/bin`, `/usr/local/bin`) et les transmet via l'option `js_runtimes` aux appels `yt-dlp` (`lister_videos_playlist` et `telecharger_video`). Attention : `yt-dlp` rejette les versions anciennes de Node (ex. Node 20 marqué `unsupported`) ; privilégier `deno`.
- **Authentification par cookies (`cookies.txt`)** : si `FICHIER_COOKIES` (`./cookies.txt`, exclu de git via `.gitignore`) existe, `_options_communes()` ajoute `"cookiefile": FICHIER_COOKIES` aux deux appels `yt-dlp`. Le fichier doit être au format Netscape HTTP Cookie File (générable via `.venv/bin/yt-dlp --cookies-from-browser <navigateur> --cookies cookies.txt --skip-download "<URL>"`).

## 7. Dépendances

Déclarées dans `requirements.txt` (versions testées) :
- `numpy`, `scipy` (audio), `yt-dlp[default]` (téléchargement + `yt-dlp-ejs`, volontairement non figé car YouTube évolue fréquemment), `opencv-python` (frames), `requests` (Ollama), `streamlit` (interface web).
- `mlx`, `mlx-vlm` avec **marqueur de plateforme** `sys_platform=="darwin" and platform_machine=="arm64"` (ignorés hors Apple Silicon).
- **Hors pip** : `ffmpeg` et `deno` (`brew install ffmpeg deno`), et le serveur **Ollama** + un modèle instruct.
- Le modèle vision `Qwen3-VL-8B-Instruct-4bit` (~5,8 Go, licence Apache 2.0 ; pic ~6,2 Go RAM sur M5 24 Go) est téléchargé au 1er run par mlx-vlm.

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
| Registre des vidéos traitées | Flux quotidien idempotent : ne pas re-télécharger/re-analyser les anciennes vidéos de la playlist. |

## 9. Points d'extension

- **Changer de modèle de légendes** : `MODELE_LLM` dans `config.py` (tout modèle Ollama instruct).
- **Ajouter/retirer une plateforme ou une langue** : `PLATEFORMES` / `LANGUES_LEGENDES` (+ ajuster le schéma dans `_schema_enrichissement` si nouvelle plateforme).
- **Remplacer le modèle de validation visuelle** : ajuster `MODELE_VISION` dans `config.py` (tout VLM compatible `mlx-vlm` sortant `bbox_2d`), ou réécrire `vision.py` en conservant la signature `verifier_et_cadrer_vision(video_path, candidats) -> [{debut, fin, cadrage}]`. Le reste du pipeline est inchangé.
- **Cibler un autre sport / type de contenu** : ajuster `PROMPT_DETECTION` (grounding) et le prompt d'enrichissement (`_construire_prompt_enrichissement`).

## 10. Vérification

Pas de suite de tests automatisés dans le repo. Vérifications manuelles disponibles :
```bash
python3 -m py_compile src/*.py main.py            # syntaxe de tous les modules
.venv/bin/python -c "from src.pipeline import executer_pipeline_complet"   # câblage des imports
.venv/bin/python main.py "https://www.youtube.com/watch?v=..."             # run réel sur une vidéo
```
Le pipeline a été validé de bout en bout sur un cas réel (UFC 239, Masvidal–Askren) : détection audio → 2 moments, validation visuelle → 2/2, clips 404×720 rendus, légendes bilingues et manifeste générés.
