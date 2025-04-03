import cv2
import os
import uuid
import time

TEMP_IMAGE_DIR = "temp_images"

def setup_temp_dir():
    """Crée le répertoire temporaire s'il n'existe pas."""
    if not os.path.exists(TEMP_IMAGE_DIR):
        try:
            os.makedirs(TEMP_IMAGE_DIR)
            print(f"Répertoire temporaire '{TEMP_IMAGE_DIR}' créé.")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire temporaire '{TEMP_IMAGE_DIR}': {e}")
            # Si on ne peut pas créer le dossier, on ne pourra pas sauvegarder
            raise

def save_temp_image(frame, filename_prefix="detection"):
    """Sauvegarde la frame donnée dans un fichier JPG temporaire.

    Args:
        frame (numpy.ndarray): L'image (frame) à sauvegarder.
        filename_prefix (str): Préfixe pour le nom du fichier.

    Returns:
        str: Le chemin complet vers le fichier sauvegardé, ou None en cas d'erreur.
    """
    setup_temp_dir() # Assure que le dossier existe

    if frame is None:
        print("Erreur: Tentative de sauvegarde d'une frame None.")
        return None

    try:
        # Générer un nom de fichier unique
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8] # Court UUID pour l'unicité
        filename = f"{filename_prefix}_{timestamp}_{unique_id}.jpg"
        filepath = os.path.join(TEMP_IMAGE_DIR, filename)

        # Sauvegarder l'image
        success = cv2.imwrite(filepath, frame)

        if success:
            # print(f"Image temporaire sauvegardée: {filepath}") # Log optionnel
            return filepath
        else:
            print(f"Erreur lors de la sauvegarde de l'image vers {filepath}")
            return None
    except Exception as e:
        print(f"Exception lors de la sauvegarde de l'image temporaire: {e}")
        return None 