# Système de détection avec caméra ESP32

Ce projet utilise une caméra ESP32 connectée au port COM5 pour capturer des images, les transmettre à une application Flask, et effectuer une détection de personnes avec YOLOv8.

## Configuration requise

### Matériel
- ESP32-CAM
- Adaptateur USB-TTL pour programmer l'ESP32-CAM
- Connexion série sur le port COM5 (configurable dans app.py)

### Logiciels
- Python 3.7+ 
- PlatformIO pour compiler et téléverser le code sur l'ESP32
- Dépendances Python listées dans requirements.txt

## Installation

1. Installer les dépendances Python:
```
pip install -r requirements.txt
```

2. Télécharger le modèle YOLOv8:
```
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o yolov8n.pt
```

3. Configurer et flasher l'ESP32 avec PlatformIO:
```
platformio run --target upload
```

## Utilisation

1. Connecter l'ESP32-CAM au port série COM5
2. Lancer l'application Flask:
```
python app.py
```
3. Accéder à l'interface web:
   - Vue brute: http://localhost:5000/
   - Vue avec détection: http://localhost:5000/processed

## Architecture

- `esp32_camera.ino`: Programme pour l'ESP32-CAM qui capture les images et les envoie via série
- `app.py`: Application Flask qui reçoit les images de l'ESP32 et effectue la détection
- `templates/`: Contient les fichiers HTML pour l'interface web

## Notes

- Le format de données entre l'ESP32 et l'application Flask utilise un protocole personnalisé avec des marqueurs de début/fin (0xFFD8/0xFFD9) et une longueur explicite pour s'assurer de l'intégrité des données.
- La détection est limitée aux personnes (classe 0 dans COCO) mais peut être étendue à d'autres classes en modifiant le paramètre `classes` dans la fonction `detect_with_yolov8`. 