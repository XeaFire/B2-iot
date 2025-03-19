from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import os
import threading
import subprocess

# Import pour YOLOv8
try:
    from ultralytics import YOLO
    YOLOV8_AVAILABLE = True
    print("Ultralytics (YOLOv8) importé avec succès")
except ImportError:
    YOLOV8_AVAILABLE = False
    print("ERREUR: Ultralytics (YOLOv8) n'est pas installé. Veuillez l'installer avec 'pip install ultralytics'")

app = Flask(__name__)

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
        print("Initialisation de la caméra...")
        
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if camera is not None and camera.isOpened():
            # Configuration de la caméra
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            print(f"Caméra ouverte avec succès")
            self.camera = camera
            # Lire quelques frames pour stabiliser la caméra
            for _ in range(5):
                ret, _ = self.camera.read()
        else:
            print("Échec d'ouverture de la caméra")
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

# Cache pour stocker la dernière détection
last_detection = {
    'frame': None,
    'time': 0
}

def detect_with_yolov8(frame):
    global last_detection, yolo_model
    current_time = time.time()

    # Si YOLOv8 n'est pas disponible, renvoyer la frame sans traitement
    if yolo_model is None:
        # Ajouter un texte explicatif
        cv2.putText(frame, "YOLOv8 non disponible", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    # Vérifier si on doit réutiliser la dernière détection
    if current_time - last_detection['time'] < detection_interval and last_detection['frame'] is not None:
        return last_detection['frame'].copy()
    
    # Faire une nouvelle détection
    if frame is None or frame.size == 0:
        print("Frame invalide passée à detect_with_yolov8")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Créer une copie de la frame pour le traitement
    processed_frame = frame.copy()
    
    try:
        # Mesurer le temps de détection
        start_time = time.time()
        
        # Effectuer la détection avec YOLOv8
        results = yolo_model(processed_frame, conf=0.3, classes=[0])  # Classe 0 = personne
        
        # Traiter les résultats
        if results and len(results) > 0:
            # Dessiner les résultats sur la frame
            annotated_frame = results[0].plot()
            
            # Ajouter l'information de FPS
            detection_time = time.time() - start_time
            fps_text = f"Detection: {1/detection_time:.1f} FPS"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mettre à jour le cache
            last_detection['frame'] = annotated_frame
            last_detection['time'] = current_time
            
            return annotated_frame
        else:
            # Si pas de détection, ajouter juste l'info FPS
            detection_time = time.time() - start_time
            fps_text = f"Detection: {1/detection_time:.1f} FPS"
            cv2.putText(processed_frame, fps_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mettre à jour le cache
            last_detection['frame'] = processed_frame
            last_detection['time'] = current_time
            
            return processed_frame
            
    except Exception as e:
        print(f"Erreur lors de la détection YOLOv8: {e}")
        # Ajouter un message d'erreur
        cv2.putText(processed_frame, f"Erreur: {str(e)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return processed_frame

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
    # Obtenir l'instance de la caméra
    camera_manager = CameraManager.get_instance()
    
    while True:
        ret, frame = camera_manager.get_frame()
        
        if ret:
            # Appliquer la détection YOLOv8
            frame = detect_with_yolov8(frame)
        
        # Encoder et renvoyer la frame avec qualité réduite pour la performance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.02)  # 50 FPS pour la fluidité

@app.route('/')
def index():
    # Page simple pour afficher le flux brut
    return render_template('raw.html')

@app.route('/raw_feed')
def raw_feed():
    return Response(gen_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed')
def processed():
    return render_template('processed.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Lancement du serveur Flask...")
    try:
        app.run(debug=True, host='0.0.0.0')
    finally:
        # Libérer la caméra à la fermeture
        camera_manager = CameraManager.get_instance()
        camera_manager.release()
        print("Application terminée, ressources libérées")
