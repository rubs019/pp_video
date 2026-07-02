"""Pipeline de génération de highlights verticaux à partir de vidéos de combats.

Découpé en modules : config, ffmpeg_utils, telechargement, audio (étape 1),
vision (étape 2), decoupage (étape 3), enrichissement (étape 4), pipeline (orchestration).
Point d'entrée : main.py à la racine -> src.pipeline.executer_pipeline_complet.
"""
