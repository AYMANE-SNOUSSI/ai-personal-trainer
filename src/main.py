import cv2
import numpy as np
import time
import winsound 
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'models/yolov8n-pose.pt' 

# 1. SEUILS (Assouplis pour être plus permissifs sur les genoux)
ARM_ANGLE_DOWN = 100    # Bras plié (Bas)
ARM_ANGLE_UP = 150      # Bras tendu (Haut)

# Pour le dos, on vise 180°. On accepte entre 150° et 210°
ANGLE_DOS_MIN = 150
ANGLE_DOS_MAX = 210

# Couleurs
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)

def calculate_angle(a, b, c):
    """Calcule l'angle au point b."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: 
        angle = 360 - angle
    return angle

# --- INITIALISATION ---
print("Chargement du modèle...")
model = YOLO(MODEL_PATH)

print("Ouverture de la caméra 2 (DSHOW)...")
# TA CONFIGURATION SPÉCIFIQUE
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# On force une bonne résolution pour aider la détection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERREUR : Impossible d'ouvrir la caméra 2. Vérifie DroidCam.")
    exit()

counter = 0
stage = "UP" 
feedback_rep = ""
feedback_timer = 0
DISPLAY_WIDTH = 1024 

print("SYSTÈME PRÊT ! L'IA choisira automatiquement le meilleur profil.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Redimensionnement
    height, width = frame.shape[:2]
    ratio = DISPLAY_WIDTH / width
    new_height = int(height * ratio)
    frame = cv2.resize(frame, (DISPLAY_WIDTH, new_height))

    # Détection IA
    results = model(frame, verbose=False)

    # Focus sur la plus grande personne
    keypoints_data = None
    max_area = 0
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                # On récupère TOUTES les infos (x, y, confiance)
                keypoints_data = results[0].keypoints.data[i].cpu().numpy()

    if keypoints_data is not None:
        # keypoints_data est de forme (17, 3) -> [x, y, conf]
        
        # --- ETAPE 1 : CHOIX DU COTÉ (GAUCHE vs DROIT) ---
        # Indices YOLO:
        # Gauche (Left): Epaule=5, Coude=7, Poignet=9, Hanche=11, Genou=13
        # Droit (Right): Epaule=6, Coude=8, Poignet=10, Hanche=12, Genou=14
        
        # On compare la confiance des épaules pour savoir quel côté est visible
        conf_left = keypoints_data[5][2]
        conf_right = keypoints_data[6][2]
        
        if conf_right > conf_left:
            side = "DROIT"
            # On utilise les indices pairs
            idx_s, idx_e, idx_w, idx_h, idx_k = 6, 8, 10, 12, 14
        else:
            side = "GAUCHE"
            # On utilise les indices impairs
            idx_s, idx_e, idx_w, idx_h, idx_k = 5, 7, 9, 11, 13

        # Extraction des coordonnées (x,y)
        p_shoulder = keypoints_data[idx_s][:2]
        p_elbow    = keypoints_data[idx_e][:2]
        p_wrist    = keypoints_data[idx_w][:2]
        p_hip      = keypoints_data[idx_h][:2]
        p_knee     = keypoints_data[idx_k][:2]

        # Vérification qu'on voit bien les points essentiels (confiance > 0.5)
        if keypoints_data[idx_s][2] > 0.5 and keypoints_data[idx_h][2] > 0.5 and keypoints_data[idx_k][2] > 0.5:
            
            # --- ETAPE 2 : CALCUL POSTURE DOS (CRITIQUE) ---
            # On calcule UNIQUEMENT Epaule - Hanche - Genou
            # C'est valide pour les pompes pieds ET genoux
            angle_back = calculate_angle(p_shoulder, p_hip, p_knee)
            
            # Est-ce que le dos est droit ?
            if ANGLE_DOS_MIN < angle_back < ANGLE_DOS_MAX:
                posture_correcte = True
                color_spine = GREEN
                msg_dos = "DOS OK"
            else:
                posture_correcte = False
                color_spine = RED
                msg_dos = "DOS !" # Dos rond ou creusé

            # --- ETAPE 3 : COMPTEUR ---
            # On vérifie si on voit le bras pour compter
            if keypoints_data[idx_e][2] > 0.5 and keypoints_data[idx_w][2] > 0.3:
                angle_arm = calculate_angle(p_shoulder, p_elbow, p_wrist)
                
                # Debug visuel angle bras
                cv2.putText(frame, f"{int(angle_arm)}", (int(p_elbow[0]), int(p_elbow[1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

                # Logique UP / DOWN
                if angle_arm < ARM_ANGLE_DOWN: 
                    stage = "DOWN" # On est en bas
                
                if angle_arm > ARM_ANGLE_UP and stage == "DOWN":
                    # On remonte !
                    if posture_correcte:
                        counter += 1
                        feedback_rep = "GOOD !"
                        # Petit son aigu
                        try: winsound.Beep(1500, 100) 
                        except: pass
                    else:
                        feedback_rep = "MAUVAISE FORME"
                        # Son grave
                        try: winsound.Beep(400, 300) 
                        except: pass
                    
                    stage = "UP"
                    feedback_timer = time.time()

                # Dessin Bras
                color_arm = BLUE if stage == "UP" else ORANGE
                cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_elbow[0]), int(p_elbow[1])), color_arm, 4)
                cv2.line(frame, (int(p_elbow[0]), int(p_elbow[1])), (int(p_wrist[0]), int(p_wrist[1])), color_arm, 4)

            # --- AFFICHAGE ---
            # Ligne du dos (Epaule -> Hanche -> Genou)
            # On ne dessine PAS vers la cheville pour éviter la confusion visuelle
            cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_hip[0]), int(p_hip[1])), color_spine, 4)
            cv2.line(frame, (int(p_hip[0]), int(p_hip[1])), (int(p_knee[0]), int(p_knee[1])), color_spine, 4)

            # Info Texte sur la personne
            cv2.putText(frame, msg_dos, (int(p_hip[0]), int(p_hip[1]-40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_spine, 2)
            # Indicateur du côté détecté (Debug)
            cv2.putText(frame, f"Cote: {side}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # --- HUD (Affichage Fixe) ---
    cv2.rectangle(frame, (0,0), (350, 120), (0,0,0), -1)
    cv2.putText(frame, str(counter), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
    cv2.putText(frame, f"REPS ({stage})", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # Feedback temporaire
    if time.time() - feedback_timer < 2 and feedback_rep != "":
        col = GREEN if feedback_rep == "GOOD !" else RED
        cv2.putText(frame, feedback_rep, (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 3)

    cv2.imshow("Coach IA - V5 (Auto-Side)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()