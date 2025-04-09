# Configuration de l'application

# Mode développeur
# True: Affiche les informations de débogage (ex: ROI, boîtes YOLO) sur le flux vidéo web.
# False: Affiche un flux vidéo plus propre sans les superpositions de débogage.
DEV_MODE = True  # Mettez à False pour la production/démonstration 

# Source de la caméra
# Mettre un entier (ex: 0, 1) pour une caméra locale connectée au PC.
# Mettre une URL (chaîne de caractères, ex: 'http://192.168.1.10:8080/video') 
# pour un flux vidéo réseau (IP Camera, autre PC, etc.).
CAMERA_SOURCE = 0 # 0 pour la webcam par défaut s