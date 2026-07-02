"""Point d'entrée du pipeline de highlights.

Usage :
    .venv/bin/python main.py "https://www.youtube.com/playlist?list=..."   # URL en argument
    .venv/bin/python main.py "https://www.youtube.com/watch?v=..."          # une vidéo isolée
    .venv/bin/python main.py                                                 # défaut : URL_SOURCE (src/config.py)
"""
import argparse

from src.config import URL_SOURCE
from src.pipeline import executer_pipeline_complet


def main():
    parser = argparse.ArgumentParser(
        description="Génère des highlights verticaux à partir d'une playlist ou d'une vidéo YouTube.",
    )
    parser.add_argument(
        "url", nargs="?", default=URL_SOURCE,
        help="URL de la playlist ou de la vidéo YouTube. "
             "Par défaut : URL_SOURCE défini dans src/config.py.",
    )
    args = parser.parse_args()

    if "ID_DE_LA_PLAYLIST" in args.url:
        parser.error(
            "aucune URL fournie et URL_SOURCE (src/config.py) est encore le gabarit. "
            'Passez une URL en argument : python main.py "https://www.youtube.com/..."'
        )

    executer_pipeline_complet(args.url)


if __name__ == "__main__":
    main()
