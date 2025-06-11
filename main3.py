import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
import warnings
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
warnings.filterwarnings("ignore", category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)


# Load models
face_model = YOLO("../models/yolov8n-face.pt")
emotion_model = load_model("../mini_pro_integrate/models/emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mediapipe face mesh for landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# Drowsiness thresholds
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 20
COUNTER = 0

# Indices for eyes landmarks (from MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    eye = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

def detect_expression(landmarks, image_w, image_h):
    # Get relevant points
    mouth_top = landmarks[13]  # upper inner lip
    mouth_bottom = landmarks[14]  # lower inner lip
    mouth_left = landmarks[78]
    mouth_right = landmarks[308]

    # Convert to pixel
    top = np.array([mouth_top.x * image_w, mouth_top.y * image_h])
    bottom = np.array([mouth_bottom.x * image_w, mouth_bottom.y * image_h])
    left = np.array([mouth_left.x * image_w, mouth_left.y * image_h])
    right = np.array([mouth_right.x * image_w, mouth_right.y * image_h])

    mouth_vert = np.linalg.norm(top - bottom)
    mouth_horiz = np.linalg.norm(left - right)

    mouth_ratio = mouth_vert / mouth_horiz

    if mouth_ratio > 0.5:
        return "Yawn"
    elif mouth_ratio > 0.3:
        return "Smile"
    else:
        return "Neutral"

def detect_emotion(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = emotion_model.predict(face, verbose=0)[0]
    return emotion_labels[np.argmax(preds)], np.max(preds)

# Start camera
cap = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_h, image_w = frame.shape[:2]
    results = face_model(frame)[0]

    for det in results.boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Emotion detection
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        emotion, prob = detect_emotion(gray)
        cv2.putText(frame, f"{emotion} ({prob:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Drowsiness detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:

                # mp_drawing.draw_landmarks(
                # image=frame,
                # landmark_list=landmarks,
                # connections=mp_face_mesh.FACEMESH_TESSELATION,
                # landmark_drawing_spec=None,
                # connection_drawing_spec=mp_drawing_styles
                # .get_default_face_mesh_tesselation_style()
        # )

                leftEAR = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, image_w, image_h)
                rightEAR = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, image_w, image_h)
                ear = (leftEAR + rightEAR) / 2.0

                # Detect Expression from face geometry
                expression = detect_expression(landmarks.landmark, image_w, image_h)

                # Draw expression on screen
                cv2.putText(frame, f"Expression: {expression}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSY ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    COUNTER = 0

    cv2.imshow("Multi-Utility Human Behavior Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)