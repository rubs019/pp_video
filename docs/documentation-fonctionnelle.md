# Documentation fonctionnelle — Machine à highlights

## 1. Objectif

Transformer automatiquement des vidéos de combats (UFC / MMA / boxe) publiées sur YouTube en **clips courts verticaux (format 9:16)** prêts à publier sur les réseaux sociaux, **accompagnés de tout le nécessaire pour la publication** : titre lisible, légendes bilingues, hashtags, et un tableau de pilotage.

Le pipeline répond à une chaîne complète : « je donne une playlist (ou une vidéo) → je récupère des extraits prêts à poster, correctement identifiés et légendés ».

## 2. Ce que fait le pipeline, étape par étape (vue métier)

Pour chaque vidéo traitée :

1. **Téléchargement** — la vidéo est récupérée depuis YouTube (qualité plafonnée à 720p pour la rapidité).
2. **Repérage des moments forts (audio)** — le pipeline « écoute » la bande-son et repère les pics d'intensité sonore (la foule qui s'enflamme, un KO). Ce sont les candidats highlights.
3. **Validation à l'image (IA visuelle)** — chaque moment repéré est confirmé visuellement : on vérifie que des combattants sont bien à l'image, et on **recentre automatiquement** le cadrage vertical sur l'action. Un pic sonore sans action visible est écarté (évite les faux positifs).
4. **Découpe & recadrage** — les segments validés sont découpés et recadrés en 9:16 vertical.
5. **Enrichissement (IA de texte, en local)** — le pipeline lit le titre/description de la vidéo et en déduit les métadonnées (combattants, événement, date du combat, catégorie de poids, résultat), puis **rédige des légendes prêtes à publier**.

Une vidéo « aveugle » en entrée ressort donc en **plusieurs clips identifiés, cadrés et légendés**.

## 3. Ce que vous obtenez en sortie

Tout arrive dans le dossier `highlights_prets/`. Pour chaque clip :

| Sortie | Description |
|--------|-------------|
| **Le clip vidéo** | MP4 vertical 9:16, ~20 s, nommé lisiblement. Ex. : `2019-07-06_ufc-239_masvidal-vs-askren_ko_0.mp4` |
| **Un fichier légende `.txt` par vidéo** | Nommé comme la vidéo (`{base_nom}.txt`), il couvre **tous ses clips** et liste leurs noms. Contient, pour **4 plateformes** (Instagram Reels, TikTok, YouTube Shorts, X) et en **français + anglais**, un texte accrocheur et des hashtags — prêts à copier-coller. |
| **Le manifeste** (`manifeste.csv` et `manifeste.json`) | Un **tableau récapitulatif de tous les clips** produits : fichier, combattants, vainqueur/perdant, événement, date du combat, catégorie de poids, méthode de victoire, chemin de la légende. |

### Exemple concret (test réel)
Entrée : une vidéo « Jorge Masvidal vs Ben Askren — UFC 239 ».
Sortie : 2 clips (`..._0.mp4`, `..._1.mp4`), chacun avec sa légende `.txt` bilingue, et une ligne par clip dans le manifeste avec : combattants = *Jorge Masvidal, Ben Askren*, vainqueur = *Masvidal*, événement = *UFC 239*, **date_combat = 2019-07-06**, poids = *Welterweight*, méthode = *KO*.

### Le manifeste : votre outil de pilotage
Le `manifeste.csv` s'ouvre dans n'importe quel tableur. C'est là que vous **planifiez vos publications** : une ligne = un clip, avec toutes les infos pour choisir quoi poster, quand, sur quelle plateforme, avec quelle légende. Il est **cumulatif** (chaque nouvelle vidéo ajoute ses lignes) et **sans doublon**.

## 4. Comment lancer

L'URL se donne **directement au lancement** (pas besoin de modifier de fichier) :

```bash
# Une playlist entière
.venv/bin/python main.py "https://www.youtube.com/playlist?list=VOTRE_PLAYLIST"

# Une seule vidéo
.venv/bin/python main.py "https://www.youtube.com/watch?v=VOTRE_VIDEO"

# Tout retraiter en ignorant le registre des vidéos déjà traitées
.venv/bin/python main.py "https://www.youtube.com/playlist?list=..." --force
```

**Interface web (Streamlit)** — pour lancer un traitement et parcourir les résultats sans ligne de commande :
```bash
.venv/bin/streamlit run interface_web.py
```
Trois onglets : **Lancer** (URL + option de retraitement, logs en direct), **Galerie** (lecture des clips + légendes FR/EN à copier), **Manifeste & registre** (tableaux, téléchargement CSV, vidéos déjà traitées).

Le pipeline traite les vidéos **une par une** (« flux tendu ») : téléchargement → traitement → suppression de la source pour économiser l'espace disque → vidéo suivante.

### Lancement quotidien : les vidéos déjà traitées sont ignorées

Le pipeline tient un **registre** (`highlights_prets/videos_traitees.json`) des vidéos déjà traitées. À chaque lancement, il **saute automatiquement** celles qu'il a déjà analysées, **avant même de les télécharger**. Vous pouvez donc **relancer sur la même playlist chaque jour** sans retraiter l'existant : seules les nouvelles vidéos sont traitées.

- Une vidéo est enregistrée dès qu'elle a été **analysée jusqu'au bout**, même si elle n'a produit aucun clip.
- Le fichier registre est lisible (ID, date de traitement, titre, nombre de clips) — c'est votre trace de « ce qui a déjà été fait ».
- Pour **forcer le retraitement** malgré le registre : ajoutez `--force`.

## 5. Ce que vous pouvez ajuster (dans `src/config.py`)

| Réglage | Effet |
|---------|-------|
| `DUREE_HIGHLIGHT` | Durée visée des clips (défaut 30 s). |
| `RECADRAGE_DYNAMIQUE` | `True` = le cadre **suit l'action** (panoramique fluide) ; `False` (défaut) = cadrage figé, plus rapide. |
| `PERCENTILE_PIC` / `MARGE_MIN_DB` | Sensibilité de la détection audio (plus le percentile est haut, moins on retient de moments). |
| `PLATEFORMES` | Liste des plateformes pour lesquelles générer des légendes. |
| `LANGUES_LEGENDES` | Langues des légendes (défaut : français + anglais). |
| `MODELE_VISION` | Modèle VLM (MLX) pour la validation visuelle (`Qwen3-VL-8B-Instruct-4bit` par défaut, ou `4B-Instruct-4bit` sur Mac 16 Go). |
| `VISION_LARGEUR_MAX` | Largeur max des frames envoyées au VLM (défaut `640` px) pour accélérer l'inférence sans perte de cadrage. |
| `MODELE_LLM` | Modèle Ollama utilisé pour les légendes. Un modèle plus léger (ex. `mistral:7b`) = plus rapide. |
| `FICHIER_COOKIES` | Chemin vers le fichier `cookies.txt` (défaut `./cookies.txt`), utilisé automatiquement par `yt-dlp` s'il existe. |

## 6. Limites connues et comportements attendus

- **Aucun clip produit** si : la bande-son n'a aucun pic marquant, ou l'IA visuelle ne détecte personne sur les moments repérés. C'est voulu (on ne publie pas du vide).
- **Vidéos mono-combat** : le pipeline suppose qu'une vidéo = un combat. Tous les clips d'une même vidéo héritent des mêmes métadonnées. Une compilation multi-combats ne serait pas correctement segmentée par combat.
- **Légendes indisponibles** : si le service de texte local (Ollama) n'est pas lancé, les clips sont **quand même produits** (nommés avec l'identifiant YouTube), mais sans légendes enrichies. Le pipeline ne s'arrête jamais pour ça.
- **Un échec sur une vidéo n'arrête pas le lot** : la vidéo fautive est signalée et le traitement passe à la suivante.

## 7. Prérequis (environnement)

- **Mac Apple Silicon (M4)** — l'étape de validation visuelle repose sur un modèle optimisé Apple Silicon.
- **ffmpeg** et **Deno** installés (`brew install ffmpeg deno`). Deno sert de runtime JavaScript pour `yt-dlp` afin de décoder les formats vidéo YouTube.
- **Ollama** lancé localement avec un modèle instruct, pour les légendes.
- Dépendances Python : `pip install -r requirements.txt` (utiliser `pip install -U -r requirements.txt` pour garder `yt-dlp[default]` à jour).

### Authentification YouTube (`cookies.txt`)

Sur certaines vidéos, YouTube bloque les requêtes anonymes (*« Sign in to confirm you're not a bot »*). Le pipeline détecte automatiquement un fichier `cookies.txt` au format Netscape à la racine du projet (ignoré par git).

Pour le générer depuis ton navigateur connecté (remplacer `brave` par `chrome`, `firefox` ou `safari` selon ton navigateur) :

```bash
.venv/bin/yt-dlp --cookies-from-browser brave --cookies cookies.txt --skip-download "https://www.youtube.com/watch?v=..."
```

Alternativement, tu peux exporter tes cookies au format Netscape avec une extension dédiée (ex. *Get cookies.txt LOCALLY*) et enregistrer le fichier à `./cookies.txt`.

## 8. Point juridique — licence modèle & droits vidéo

- **Modèles IA** : le modèle de validation visuelle (**Qwen3-VL-8B-Instruct-4bit**) est sous **licence Apache 2.0** (usage commercial autorisé).
- **Droits sur les vidéos sources** : pour toute exploitation commerciale (prestation client, diffusion monétisée), assurez-vous de traiter uniquement des vidéos dont vous ou votre client détenez les droits d'exploitation.
