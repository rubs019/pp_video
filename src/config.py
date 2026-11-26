"""Configuration centrale du pipeline : toutes les constantes ajustables sont ici."""
import os

# --- Sources / dossiers ---
URL_SOURCE = "https://www.youtube.com/playlist?list=ID_DE_LA_PLAYLIST"  # Mettre l'URL de la playlist ici
DOSSIER_BRUT = "./combats_bruts/"
DOSSIER_SORTIE = "./highlights_prets/"
FICHIER_COOKIES = "./cookies.txt"  # Utilisé par yt-dlp s'il existe (authentification YouTube)

# --- Étape 1 : détection audio ---
PERCENTILE_PIC = 98           # Top (100-X)% des fenêtres = pics candidats (auto-adaptatif à la dynamique)
MARGE_MIN_DB = 4              # Marge minimale au-dessus du niveau ambiant (anti faux positifs)
DUREE_HIGHLIGHT = 30          # Durée estimée du clip en secondes

# --- Étape 2 : validation visuelle ---
MODELE_VISION = "mlx-community/LocateAnything-3B-4bit"  # VLM de grounding (MLX / Apple Silicon)
PROMPT_DETECTION = "Detect the fighters."  # Grounding ciblé : isole les combattants (écarte arbitre/coins/public)
FRAMES_PAR_SEGMENT = 2        # Frames échantillonnées par candidat (2 = compromis vitesse/robustesse)
RATIO_VERTICAL = 9 / 16       # Format de sortie vertical (9:16)
RECADRAGE_DYNAMIQUE = True   # True -> panoramique fluide (suit l'action) ; False -> cadrage statique
INTERVALLE_SUIVI = 3          # secondes entre keyframes de suivi (clip 30 s -> ~10 keyframes)
LISSAGE_FENETRE = 3           # taille (impaire) de la moyenne glissante sur la trajectoire cx(t)
PAN_VITESSE_MAX = 0.20        # vitesse de pano max, en fraction de crop_w par seconde (anti à-coups)
SEUIL_PAN_FIT = 0.5           # fraction min de keyframes où l'envergure tient dans le crop pour paner

# --- Étape 4 : enrichissement métadonnées (LLM local Ollama) ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELE_LLM = "mistral:7b"     # instruct local (compromis vitesse/qualité) ; + rapide : "llama3.2" ; + fin : "qwen3.5:9b"
OLLAMA_TIMEOUT = 120          # secondes
OLLAMA_NUM_CTX = 8192
PLATEFORMES = ["instagram_reels", "tiktok", "youtube_shorts", "x"]
LANGUES_LEGENDES = ["fr", "en"]   # légendes bilingues
VALEUR_INCONNUE = "inconnu"
NOM_FICHIER_MAX = 80          # longueur max du base_nom slugifié
MANIFESTE_JSON = os.path.join(DOSSIER_SORTIE, "manifeste.json")  # source de vérité
MANIFESTE_CSV = os.path.join(DOSSIER_SORTIE, "manifeste.csv")    # régénéré depuis le JSON
FICHIER_HISTORIQUE = os.path.join(DOSSIER_SORTIE, "videos_traitees.json")  # registre des vidéos déjà traitées

os.makedirs(DOSSIER_BRUT, exist_ok=True)
os.makedirs(DOSSIER_SORTIE, exist_ok=True)
