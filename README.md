# Machine à highlights

Transforme automatiquement des vidéos de combats (UFC / MMA / boxe) YouTube en **clips verticaux 9:16 prêts à publier**, avec métadonnées, légendes bilingues (FR/EN) et un manifeste de pilotage.

Chaîne en 4 étapes : détection audio des moments forts → validation visuelle (IA) et recadrage → découpe 9:16 → enrichissement métadonnées & légendes (LLM local Ollama).

## Documentation

- 📘 **[Documentation fonctionnelle](docs/documentation-fonctionnelle.md)** — ce que fait le pipeline, ce qu'on obtient, comment lancer, réglages, limites.
- 🔧 **[Documentation technique](docs/documentation-technique.md)** — architecture, algorithmes, décisions de conception, points d'extension.
- 🤖 **[CLAUDE.md](CLAUDE.md)** — repères pour travailler dans le dépôt (commandes, structure, pièges).

## Démarrage rapide

```bash
# Prérequis : Mac Apple Silicon, ffmpeg (brew install ffmpeg), serveur Ollama lancé
.venv/bin/pip install -r requirements.txt

# Lancer sur une playlist ou une vidéo (URL en argument)
.venv/bin/python main.py "https://www.youtube.com/playlist?list=..."
.venv/bin/python main.py "https://www.youtube.com/watch?v=..."

# Interface web locale (Streamlit) : lancer un traitement, parcourir clips et légendes
.venv/bin/streamlit run interface_web.py
```

Les clips, légendes `.txt` et le manifeste (`manifeste.csv` / `.json`) arrivent dans `highlights_prets/`.

## Structure

```
main.py            Point d'entrée CLI (URL en argument)
interface_web.py   Interface web locale (Streamlit)
src/               Package : config, ffmpeg_utils, telechargement, audio,
                   vision, decoupage, enrichissement, pipeline
docs/              Documentation fonctionnelle et technique
requirements.txt   Dépendances Python épinglées
```

## ⚠️ Licence

Le modèle de validation visuelle **LocateAnything-3B** est sous **licence NVIDIA recherche / non-commerciale**. Toute diffusion commerciale doit d'abord être validée juridiquement — voir la [documentation technique](docs/documentation-technique.md#9-points-dextension) pour remplacer le modèle.
