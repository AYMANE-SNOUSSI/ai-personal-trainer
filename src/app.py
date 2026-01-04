import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
import winsound

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Coach IA", layout="wide")

# --- PARAMÈTRES (CONSTANTES) ---
MODEL_PATH = 'models/yolov8n-pose.pt' 
ARM_ANGLE_DOWN = 110    # Angle un peu plus large pour faciliter la détection en bas
ARM_ANGLE_UP = 150      
ANGLE_DOS_MIN = 140
ANGLE_DOS_MAX = 220

# Couleurs
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 165, 255)

# --- FONCTIONS ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- INTERFACE UTILISATEUR ---
st.title("🏋️‍♂️ Assistant Coach Sportif IA - V3 (Correctif)")

# Barre latérale
st.sidebar.header("Réglages")
cam_source = st.sidebar.radio("Source Caméra", ["Webcam (Index 2)", "Webcam (Index 0)", "DroidCam IP"])

# Curseur connecté à TOUT le code maintenant
confidence = st.sidebar.slider("Sensibilité IA (Baisser si lignes disparaissent)", 0.0, 1.0, 0.25)

start_button = st.sidebar.button("Démarrer", type="primary")
stop_button = st.sidebar.button("Arrêter")

col_video, col_stats = st.columns([3, 1])

with col_stats:
    st.markdown("### 📊 Performances")
    # Plus de message "En Pause", direct le compteur
    kpi_counter = st.empty()
    kpi_status = st.empty()
    kpi_feedback = st.empty()
    st.markdown("---")
    kpi_debug = st.empty()

with col_video:
    frame_placeholder = st.empty()

# --- BOUCLE PRINCIPALE ---
if start_button:
    model = load_model()
    
    if cam_source == "Webcam (Index 2)":
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    elif cam_source == "Webcam (Index 0)":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('http://192.168.1.XX:4747/video')

    counter = 0
    stage = "UP"
    posture_correcte = True
    feedback_message = ""
    feedback_timer = 0

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur de lecture caméra.")
            break

        # Analyse IA
        results = model(frame, verbose=False, conf=confidence)
        
        keypoints_data = None
        max_area = 0
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    keypoints_data = results[0].keypoints.data[i].cpu().numpy()

        if keypoints_data is not None:
            # Choix côté
            conf_left = keypoints_data[5][2]
            conf_right = keypoints_data[6][2]
            
            if conf_right > conf_left:
                side = "DROIT"
                idx_s, idx_e, idx_w, idx_h, idx_k = 6, 8, 10, 12, 14
            else:
                side = "GAUCHE"
                idx_s, idx_e, idx_w, idx_h, idx_k = 5, 7, 9, 11, 13

            p_shoulder = keypoints_data[idx_s][:2]
            p_elbow    = keypoints_data[idx_e][:2]
            p_wrist    = keypoints_data[idx_w][:2]
            p_hip      = keypoints_data[idx_h][:2]
            p_knee     = keypoints_data[idx_k][:2]
            
            # --- CORRECTION MAJEURE ICI ---
            # On utilise la variable 'confidence' du curseur au lieu de 0.5 fixe
            seuil_detection = confidence 

            # On vérifie si on voit Épaule, Hanche, Genou
            if keypoints_data[idx_s][2] > seuil_detection and \
               keypoints_data[idx_h][2] > seuil_detection and \
               keypoints_data[idx_k][2] > seuil_detection:
                
                # 1. DOS
                angle_back = calculate_angle(p_shoulder, p_hip, p_knee)
                if ANGLE_DOS_MIN < angle_back < ANGLE_DOS_MAX:
                    posture_correcte = True
                    color_spine = GREEN
                    status_msg = "DOS OK"
                else:
                    posture_correcte = False
                    color_spine = RED
                    status_msg = "DOS CREUSÉ !"

                # 2. COMPTEUR (Plus de blocage "Debout")
                # On vérifie juste si on voit le bras
                if keypoints_data[idx_e][2] > seuil_detection and keypoints_data[idx_w][2] > seuil_detection:
                    angle_arm = calculate_angle(p_shoulder, p_elbow, p_wrist)
                    
                    # LOGIQUE HAUT/BAS
                    if angle_arm < ARM_ANGLE_DOWN:
                        stage = "DOWN"
                    
                    if angle_arm > ARM_ANGLE_UP and stage == "DOWN":
                        if posture_correcte:
                            counter += 1
                            feedback_message = "GOOD REP! ✅"
                            try: winsound.Beep(1500, 100) 
                            except: pass
                        else:
                            feedback_message = "MAUVAISE FORME ❌"
                            try: winsound.Beep(400, 300) 
                            except: pass
                        
                        stage = "UP"
                        feedback_timer = time.time()

                    # Dessin Bras
                    color_arm = BLUE if stage == "UP" else ORANGE
                    cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_elbow[0]), int(p_elbow[1])), color_arm, 4)
                    cv2.line(frame, (int(p_elbow[0]), int(p_elbow[1])), (int(p_wrist[0]), int(p_wrist[1])), color_arm, 4)
                    
                    # Debug Angle
                    kpi_debug.info(f"Angle Bras: {int(angle_arm)}°")

                # Dessin Dos
                cv2.line(frame, (int(p_shoulder[0]), int(p_shoulder[1])), (int(p_hip[0]), int(p_hip[1])), color_spine, 4)
                cv2.line(frame, (int(p_hip[0]), int(p_hip[1])), (int(p_knee[0]), int(p_knee[1])), color_spine, 4)

                # UPDATE INTERFACE
                kpi_counter.metric(label="Répétitions", value=counter, delta=stage)
                if posture_correcte:
                    kpi_status.success(f"Posture : {status_msg}")
                else:
                    kpi_status.error(f"Posture : {status_msg}")

        if time.time() - feedback_timer < 2 and feedback_message != "":
            kpi_feedback.markdown(f"## {feedback_message}")
        else:
            kpi_feedback.empty()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()