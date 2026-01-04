import cv2
import numpy as np
from ultralytics import YOLO
import imageio # librairie pour le GIF
import os

# --- CONFIGURATION ---
VIDEO_INPUT = 'data/test1_pushup.mp4'  
MODEL_PATH = 'models/yolov8n-pose.pt'

# Noms des fichiers de sortie
OUTPUT_VIDEO = 'output_result.mp4'
OUTPUT_GIF = 'demo_coach.gif'

# Paramètres IA
ARM_ANGLE_DOWN = 100    
ARM_ANGLE_UP = 150      
ANGLE_DOS_MIN = 140
ANGLE_DOS_MAX = 220
CONFIDENCE_THRESHOLD = 0.5

# Couleurs
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- INITIALISATION ---
print(f"Chargement du modèle : {MODEL_PATH}")
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_INPUT)

# --- CONFIGURATION SAUVEGARDE VIDÉO ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# On redimensionne pour l'affichage et le GIF (pour que ce ne soit pas trop lourd)
DISPLAY_WIDTH = 640
ratio = DISPLAY_WIDTH / width
new_height = int(height * ratio)

# Initialisation du writer (mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (DISPLAY_WIDTH, new_height))

# Liste pour stocker les images du GIF
gif_frames = []

counter = 0
stage = "UP" 
posture_correcte = True

print("Traitement en cours... (Cela peut prendre un peu de temps)")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Redimensionnement
    frame = cv2.resize(frame, (DISPLAY_WIDTH, new_height))

    # Détection
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

    keypoints_data = None
    max_area = 0
    
    # Focus sur la personne principale
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                keypoints_data = results[0].keypoints.data[i].cpu().numpy()

    if keypoints_data is not None:
        # Choix automatique du côté
        conf_left = keypoints_data[5][2]
        conf_right = keypoints_data[6][2]
        
        if conf_right > conf_left:
            idx_s, idx_e, idx_w, idx_h, idx_k = 6, 8, 10, 12, 14
        else:
            idx_s, idx_e, idx_w, idx_h, idx_k = 5, 7, 9, 11, 13

        # Extraction coordonnées
        p_shoulder = keypoints_data[idx_s][:2]
        p_elbow    = keypoints_data[idx_e][:2]
        p_wrist    = keypoints_data[idx_w][:2]
        p_hip      = keypoints_data[idx_h][:2]
        p_knee     = keypoints_data[idx_k][:2]

        # Vérification visibilité
        if keypoints_data[idx_s][2] > 0.5 and keypoints_data[idx_h][2] > 0.5 and keypoints_data[idx_k][2] > 0.5:
            
            # 1. DOS
            angle_back = calculate_angle(p_shoulder, p_hip, p_knee)
            if ANGLE_DOS_MIN < angle_back < ANGLE_DOS_MAX:
                posture_correcte = True
                color_spine = GREEN
            else:
                posture_correcte = False
                color_spine = RED

            # 2. BRAS & COMPTEUR
            if keypoints_data[idx_e][2] > 0.5 and keypoints_data[idx_w][2] > 0.3:
                angle_arm = calculate_angle(p_shoulder, p_elbow, p_wrist)
                
                if angle_arm < ARM_ANGLE_DOWN:
                    stage = "DOWN"
                
                if angle_arm > ARM_ANGLE_UP and stage == "DOWN":
                    if posture_correcte:
                        counter += 1
                    stage = "UP"

                # Dessin Bras
                color_arm = BLUE if stage == "UP" else ORANGE
                cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_elbow[0]), int(p_elbow[1])), color_arm, 4)
                cv2.line(frame, (int(p_elbow[0]), int(p_elbow[1])), (int(p_wrist[0]), int(p_wrist[1])), color_arm, 4)

            # Dessin Dos
            cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_hip[0]), int(p_hip[1])), color_spine, 4)
            cv2.line(frame, (int(p_hip[0]), int(p_hip[1])), (int(p_knee[0]), int(p_knee[1])), color_spine, 4)

    # Affichage Interface
    cv2.rectangle(frame, (0,0), (250, 80), (0,0,0), -1)
    cv2.putText(frame, str(int(counter)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv2.putText(frame, f"REPS ({stage})", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    if not posture_correcte:
        cv2.putText(frame, "DOS !", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 3)

    # 1. Sauvegarde frame dans la vidéo
    out.write(frame)

    # 2. Sauvegarde frame pour le GIF (On convertit BGR -> RGB)
    # On ne garde qu'une image sur 3 pour alléger le GIF
    if len(gif_frames) < 300: # Limite de sécurité pour pas saturer la mémoire
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)

    cv2.imshow("Traitement Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vidéo sauvegardée sous : {OUTPUT_VIDEO}")

# Création du GIF
print("Génération du GIF en cours (patience)...")
# On réduit les FPS du GIF pour qu'il ne soit pas trop lourd
imageio.mimsave(OUTPUT_GIF, gif_frames, fps=15, loop=0) 
print(f"GIF sauvegardé sous : {OUTPUT_GIF}")
print("Tu peux maintenant le glisser dans ton README GitHub !")