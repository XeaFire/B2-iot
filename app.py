from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import os
import threading
import subprocess
import queue
from bot import run_bot
from Utils.Notifier import notification_queue
from Utils.ImageManager import save_temp_image
from config import DEV_MODE, CAMERA_SOURCE
import mysql.connector
import uuid
import shutil

# --- Configuration de la base de données MySQL ---
db_config = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": os.getenv("DATABASE_PORT", 3306),
    "user": os.getenv("DATABASE_USER", "root"),
    "password": os.getenv("DATABASE_PASSWORD", ""),
    "database": os.getenv("DATABASE_NAME", "mydb"),
}

try:
    connection = mysql.connector.connect(**db_config)
    print("Database connection successful!")
except mysql.connector.Error as err:
    print(f"Error: {err}")

# Import pour YOLOv8
try:
    from ultralytics import YOLO
    YOLOV8_AVAILABLE = True
    print("Ultralytics (YOLOv8) importé avec succès")
except ImportError:
    YOLOV8_AVAILABLE = False
    print("ERREUR: Ultralytics (YOLOv8) n'est pas installé. Veuillez l'installer avec 'pip install ultralytics'")

app = Flask(__name__)

# --- Variables Globales Partagées et Verrous ---
last_processed_frame = None
processed_frame_lock = threading.Lock()

door_button_visible = False
door_button_hidden_until = 0
button_state_lock = threading.Lock()

# Log des ouvertures
LOG_IMAGE_DIR = os.path.join("static", "log_images")
MAX_LOG_ENTRIES = 100
door_opening_log = []
log_lock = threading.Lock()

# Classe singleton pour gérer la caméra
class CameraManager:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraManager()
            return cls._instance

    def __init__(self):
        self.camera = None
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.init_camera()
        # Démarrer le thread de capture
        self.capture_thread = threading.Thread(target=self.update, daemon=True)
        self.capture_thread.start()
        print("CameraManager initialisé")

    def init_camera(self):
        print(f"Initialisation de la caméra depuis la source: {CAMERA_SOURCE}...")
        
        camera = None
        # Choisir le backend en fonction du type de source
        if isinstance(CAMERA_SOURCE, int):
            print("Source locale détectée (entier), utilisation de CAP_DSHOW.")
            camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
        elif isinstance(CAMERA_SOURCE, str):
            print("Source réseau/fichier détectée (chaîne), backend automatique.")
            camera = cv2.VideoCapture(CAMERA_SOURCE)
        else:
            print(f"ERREUR: Type de CAMERA_SOURCE non supporté: {type(CAMERA_SOURCE)}")
            return # Ne pas continuer si la source n'est pas valide

        # Vérifier si l'ouverture a réussi
        if camera is not None and camera.isOpened():
            # Configuration de la caméra (optionnel, peut échouer sur les flux réseau)
            try:
                 camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                 camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                 camera.set(cv2.CAP_PROP_FPS, 30)
                 print("Configuration dimensions/FPS tentée.")
            except Exception as e:
                 print(f"Note: Impossible de définir les propriétés de la caméra (normal pour certains flux): {e}")
            
            print(f"Caméra/Flux ouvert avec succès depuis {CAMERA_SOURCE}")
            self.camera = camera
            # Lire quelques frames pour stabiliser
            for _ in range(5):
                ret, _ = self.camera.read()
        else:
            print(f"Échec d'ouverture de la caméra/flux depuis {CAMERA_SOURCE}")
            self.camera = None
    
    def update(self):
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Camera non disponible", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame_count = 0
        while self.running:
            if self.camera is None or not self.camera.isOpened():
                # Tentative de réouverture
                if self.camera is not None:
                    self.camera.release()
                self.init_camera()
                with self.lock:
                    self.frame = blank_frame.copy()
                time.sleep(1)  # Attendre avant la prochaine tentative
                continue
                
            # Lire une frame
            ret, frame = self.camera.read()
            frame_count += 1
            
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                if frame_count % 100 == 0:
                    print(f"Frame #{frame_count} capturée, taille: {frame.shape}")
            else:
                print("Échec de lecture de la caméra")
                with self.lock:
                    self.frame = blank_frame.copy()
                time.sleep(0.1)  # Pause courte en cas d'échec
                
            time.sleep(0.01)  # Limiter la cadence pour économiser les ressources
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera non disponible", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return False, blank
    
    def release(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        print("Caméra libérée")

# Configuration YOLOv8
yolov8_model_path = "yolov8n.pt"  # Modèle nano (le plus petit et plus rapide)
detection_interval = 0.1  # Intervalle entre les détections (en secondes)

# Vérifier si le modèle existe
if YOLOV8_AVAILABLE and not os.path.exists(yolov8_model_path):
    print(f"ATTENTION: Le modèle {yolov8_model_path} n'existe pas!")
    print(f"Vous pouvez le télécharger depuis: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

# Charger le modèle YOLOv8
yolo_model = None
if YOLOV8_AVAILABLE:
    try:
        print(f"Chargement du modèle YOLOv8 depuis {yolov8_model_path}...")
        yolo_model = YOLO(yolov8_model_path)
        print("Modèle YOLOv8 chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle YOLOv8: {e}")
        yolo_model = None
else:
    print("YOLOv8 n'est pas disponible - la détection ne fonctionnera pas")

# Variables pour la logique de notification
CONSECUTIVE_DETECTION_THRESHOLD = 7
HUMAN_CONFIDENCE_THRESHOLD = 0.70
NOTIFICATION_COOLDOWN = 60

consecutive_human_detections = 0
last_notification_time = 0

def detect_with_yolov8(frame):
    # Cette fonction effectue la détection et la logique de notification.
    # Elle renvoie TOUJOURS la frame annotée pour les notifications Discord.
    # Le choix d'afficher ou non les annotations sur le web est fait dans detection_worker.
    global yolo_model
    global consecutive_human_detections, last_notification_time
    current_time = time.time()

    if yolo_model is None:
        cv2.putText(frame, "YOLOv8 non disponible", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    if frame is None or frame.size == 0:
        print("Frame invalide passée à detect_with_yolov8")
        # Renvoyer une frame noire pourrait être mieux qu'un tableau vide
        return np.zeros((480, 640, 3), dtype=np.uint8) 

    processed_frame = frame.copy()
    annotated_frame = processed_frame
    detection_successful = False # Flag pour savoir si la détection a fonctionné
    
    try:
        start_time = time.time()
        results = yolo_model(processed_frame, conf=HUMAN_CONFIDENCE_THRESHOLD, classes=[0])
        detection_successful = True # La détection a été exécutée
        
        human_detected_in_frame = False
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            human_detected_in_frame = True
            annotated_frame = results[0].plot()
        
        # --- Logique de notification --- 
        if human_detected_in_frame:
            consecutive_human_detections += 1
        else:
            if consecutive_human_detections > 0:
                consecutive_human_detections = 0

        if consecutive_human_detections >= CONSECUTIVE_DETECTION_THRESHOLD:
            if current_time - last_notification_time >= NOTIFICATION_COOLDOWN:
                print(f"Conditions de notification remplies ({consecutive_human_detections} détections)! Préparation de la notification.")
                
                # Sauvegarder l'image annotée *avant* d'envoyer à la queue
                image_path = save_temp_image(annotated_frame, filename_prefix="human_detected")
                
                if image_path:
                    try:
                        # Mettre un tuple (message, chemin_image) dans la queue
                        notification_data = ("Alerte : Humain détecté sur la caméra ! http://192.168.1.183:5000", image_path)
                        notification_queue.put_nowait(notification_data)
                        print(f"Notification (avec image {os.path.basename(image_path)}) mise en file d'attente.")
                        last_notification_time = current_time 
                    except queue.Full:
                        print("AVERTISSEMENT: La file d'attente de notification est pleine.")
                        # Optionnel: Supprimer l'image si la queue est pleine ?
                        # if os.path.exists(image_path): os.remove(image_path)
                else:
                    print("Erreur: Impossible de sauvegarder l'image pour la notification.")
                    # Envoyer quand même une notif texte ?
                    # try:
                    #    notification_queue.put_nowait(("Alerte: Humain détecté (erreur sauvegarde image)!", None))
                    #    last_notification_time = current_time
                    # except queue.Full:
                    #    print("AVERTISSEMENT: La file d'attente de notification est pleine.")

        # --- Fin de la logique de notification ---

        # Ajouter l'information de FPS (même si la détection a échoué, on veut savoir)
        if detection_successful:
            detection_time = time.time() - start_time
            fps = (1 / detection_time) if detection_time > 0 else 0 
            fps_text = f"Detection: {fps:.1f} FPS"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
            
    except Exception as e:
        print(f"Erreur lors de la détection YOLOv8: {e}")
        # Essayer d'ajouter l'erreur sur la frame originale
        try:
            cv2.putText(processed_frame, f"Erreur YOLO: {str(e)[:30]}...", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as overlay_error:
            print(f"Erreur lors de l'ajout du texte d'erreur sur l'image: {overlay_error}")
        return processed_frame

# --- Thread de Détection en Arrière-plan ---
def detection_worker():
    global last_processed_frame, door_button_visible, door_button_hidden_until
    print(f"Thread de détection démarré. DEV_MODE: {DEV_MODE}")
    camera_manager = CameraManager.get_instance()
    
    while True:
        ret, frame = camera_manager.get_frame()
        
        if not ret or frame is None:
            time.sleep(0.5)
            continue
            
        # 2. Exécuter la détection
        annotated = detect_with_yolov8(frame.copy())
        
        # 3. Mettre à jour la frame pour le flux web selon DEV_MODE
        with processed_frame_lock:
            if DEV_MODE:
                # En mode DEV, montrer la frame avec les annotations
                last_processed_frame = annotated.copy()
            else:
                # Hors mode DEV, montrer la frame originale sans annotations
                last_processed_frame = frame.copy() 
            
        # 4. Logique de visibilité du bouton (après la logique de notification dans detect_with_yolov8)
        # Accéder à la variable globale (déjà modifiée dans detect_with_yolov8)
        current_consecutive_detections = consecutive_human_detections
        
        with button_state_lock:
            current_time = time.time()
            # Rendre visible si seuil atteint ET délai de masquage écoulé
            if current_consecutive_detections >= CONSECUTIVE_DETECTION_THRESHOLD and current_time >= door_button_hidden_until:
                if not door_button_visible: # Log seulement si changement d'état
                    print("Conditions remplies: Rendre le bouton visible.")
                door_button_visible = True
            # Note: Le bouton n'est rendu invisible que par l'action de clic via /control/door
            # ou si le serveur redémarre (initialisé à False).
            
        time.sleep(0.05)

# Afficher les caméras disponibles (Windows seulement)
try:
    if os.name == 'nt':  # Windows
        result = subprocess.run(['powershell', '-Command', "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Caption -like '*camera*' -or $_.Caption -like '*webcam*' } | Select-Object Caption"], 
                             capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("Caméras détectées par le système :")
            for line in result.stdout.strip().split('\n'):
                if line and not line.startswith("Caption") and not line.startswith("-------"):
                    print(f"- {line.strip()}")
        else:
            print("Impossible de lister les caméras via PowerShell")
except Exception as e:
    print(f"Erreur lors de la détection des caméras : {e}")

# Fonction simple pour générer le flux vidéo brut
def gen_raw_frames():
    # Obtenir l'instance de la caméra
    camera_manager = CameraManager.get_instance()
    
    while True:
        ret, frame = camera_manager.get_frame()
        
        # Encoder et renvoyer la frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.02)  # 50 FPS max

# Fonction pour générer le flux vidéo avec détection
def gen_processed_frames():
    # Lit la dernière frame traitée par le worker thread
    global last_processed_frame
    print("Client connecté au flux traité.")
    while True:
        frame_to_send = None
        with processed_frame_lock:
            if last_processed_frame is not None:
                frame_to_send = last_processed_frame.copy()
            else:
                # Afficher une image d'attente si aucune frame n'a encore été traitée
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Attente de la premiere frame...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame_to_send = blank
                
        if frame_to_send is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Contrôler le débit du flux envoyé au client
        time.sleep(0.04) # ~25 FPS pour le flux

@app.route('/')
def index():
    # Affiche la nouvelle page principale moderne
    return render_template('index.html')

@app.route('/raw_feed')
def raw_feed():
    return Response(gen_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed')
def processed():
    return render_template('processed.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Fonctions Utilitaires ---
def setup_log_dir():
    """Crée le répertoire pour les images de log s'il n'existe pas."""
    if not os.path.exists(LOG_IMAGE_DIR):
        try:
            os.makedirs(LOG_IMAGE_DIR)
            print(f"Répertoire de logs '{LOG_IMAGE_DIR}' créé.")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire '{LOG_IMAGE_DIR}': {e}")

# Nouvelle route pour le contrôle de la porte
@app.route('/control/door', methods=['POST'])
def control_door():
    global door_button_visible, door_button_hidden_until, door_opening_log
    
    current_time_for_log = time.time()
    log_image_filename = None
    log_image_relative_path = None
    frame_to_log = None
    
    # Essayer de capturer l'image juste avant de cacher le bouton
    with processed_frame_lock:
        if last_processed_frame is not None:
            frame_to_log = last_processed_frame.copy()

    if frame_to_log is not None:
        setup_log_dir() # S'assurer que le dossier existe
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time_for_log))
            unique_id = str(uuid.uuid4())[:4]
            log_image_filename = f"log_{timestamp_str}_{unique_id}.jpg"
            log_image_full_path = os.path.join(LOG_IMAGE_DIR, log_image_filename)
            success = cv2.imwrite(log_image_full_path, frame_to_log)
            if success:
                log_image_relative_path = os.path.join("log_images", log_image_filename).replace("\\", "/") # Chemin relatif pour URL
                print(f"Image de log sauvegardée: {log_image_relative_path}")
            else:
                print(f"Erreur lors de la sauvegarde de l'image de log: {log_image_full_path}")
        except Exception as e:
            print(f"Exception lors de la sauvegarde de l'image de log: {e}")

    # Mettre à jour l'état du bouton et ajouter au log
    with button_state_lock:
        door_button_visible = False
        door_button_hidden_until = current_time_for_log + 60
        print(f"SIGNAL: Porte contrôlée. Bouton caché jusqu'à {time.strftime('%H:%M:%S', time.localtime(door_button_hidden_until))}")

        # Ajouter au log (même si l'image n'a pas pu être sauvée)
        with log_lock:
            log_entry = {
                "timestamp": current_time_for_log,
                "image_path": log_image_relative_path # Sera None si erreur
            }
            door_opening_log.insert(0, log_entry) # Ajouter au début
            # Limiter la taille du log
            if len(door_opening_log) > MAX_LOG_ENTRIES:
                # Optionnel: Supprimer l'image la plus ancienne si elle existe
                oldest_entry = door_opening_log.pop() # Enlever la fin
                # if oldest_entry.get("image_path"):
                    # old_img_path = os.path.join("static", oldest_entry["image_path"])
                    # if os.path.exists(old_img_path): 
                    #    try: os.remove(old_img_path)
                    #    except OSError as e: print(f"Could not remove old log image {old_img_path}: {e}")
    
    # Placeholder: Logique d'envoi de signal réelle ici...
    
    return jsonify({"status": "success", "message": "Signal porte envoyé, bouton caché."}), 200

# Nouvelle route pour obtenir l'état du bouton
@app.route('/button_status')
def get_button_status():
    with button_state_lock:
        # print(f"DEBUG: /button_status requested, visible: {door_button_visible}") # Log de debug si besoin
        return jsonify({"visible": door_button_visible})

# Nouvelle route pour récupérer les logs
@app.route('/logs')
def get_logs():
    with log_lock:
        # Renvoyer une copie pour éviter les modifications concurrentes pendant l'itération
        return jsonify(list(door_opening_log))

if __name__ == '__main__':
    print("Initialisation de l'application...")
    
    # Initialiser CameraManager (démarre son thread de capture)
    camera_manager = CameraManager.get_instance()
    
    # Démarrer le bot Discord dans un thread séparé
    print("Démarrage du bot Discord dans un thread...")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Démarrer le worker de détection dans un thread séparé
    print("Démarrage du worker de détection dans un thread...")
    detection_thread = threading.Thread(target=detection_worker, daemon=True)
    detection_thread.start()
    
    # Laisser un peu de temps aux threads pour démarrer
    time.sleep(5) 
    
    print("Démarrage du serveur Flask...")
    try:
        # Lancer Flask (bloquant jusqu'à l'arrêt)
        # Note: debug=True recharge le code mais peut causer des problèmes avec les threads
        # Il est préférable de le désactiver (False) pour un fonctionnement stable.
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    finally:
        # Libérer la caméra à la fermeture (si Flask s'arrête proprement)
        print("Arrêt de Flask, libération des ressources...")
        camera_manager.release()
        print("Application terminée.")
        # Note: Le thread du bot (daemon) s'arrêtera automatiquement avec le processus principal.
