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
import base64
import threading
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
warnings.filterwarnings("ignore", category=UserWarning)
absl.logging.set_verbosity(absl.logging.ERROR)

app = Flask(__name__, static_folder='./build', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
face_model = YOLO("../models/yolov8n-face.pt")
emotion_model = load_model("../mini_pro_integrate/models/emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mediapipe face mesh for landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Drowsiness thresholds
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 20
COUNTER = 0

# Indices for eyes landmarks (from MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Global variables for detection state
detection_active = False
camera_thread = None
cap = None
frame_buffer = None
show_face_markings = False  # New variable to control face markings display
detection_results = {
    "emotion": "None",
    "emotion_prob": 0,
    "expression": "None",
    "drowsy": False,
    "show_face_markings": False  # Add to detection results
}

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

def process_frame(frame):
    global COUNTER, detection_results, show_face_markings
    
    # Scale down the frame for faster processing
    scale_factor = 0.75  # Process at 75% of original size
    if scale_factor < 1.0:
        frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    else:
        frame_small = frame
    
    image_h, image_w = frame.shape[:2]
    small_h, small_w = frame_small.shape[:2]
    
    # Run detection on smaller frame
    results = face_model(frame_small)[0]
    
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    
    for det in results.boxes:
        # Get coordinates from the smaller frame
        x1_small, y1_small, x2_small, y2_small = map(int, det.xyxy[0])
        
        # Scale coordinates back to original frame size
        if scale_factor < 1.0:
            x1 = int(x1_small / scale_factor)
            y1 = int(y1_small / scale_factor)
            x2 = int(x2_small / scale_factor)
            y2 = int(y2_small / scale_factor)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, min(x1, image_w-1))
            y1 = max(0, min(y1, image_h-1))
            x2 = max(0, min(x2, image_w-1))
            y2 = max(0, min(y2, image_h-1))
        else:
            x1, y1, x2, y2 = x1_small, y1_small, x2_small, y2_small
            
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            continue
            
        # Draw face rectangle
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Emotion detection
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        emotion, prob = detect_emotion(gray)
        detection_results["emotion"] = emotion
        detection_results["emotion_prob"] = float(prob)
        
        cv2.putText(display_frame, f"{emotion} ({prob:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Drowsiness detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                leftEAR = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, image_w, image_h)
                rightEAR = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, image_w, image_h)
                ear = (leftEAR + rightEAR) / 2.0

                # Detect Expression from face geometry
                expression = detect_expression(landmarks.landmark, image_w, image_h)
                detection_results["expression"] = expression

                # Draw expression on screen
                cv2.putText(display_frame, f"Expression: {expression}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw face landmarks if enabled
                if show_face_markings:
                    mp_drawing = mp.solutions.drawing_utils
                    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))
                    mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    
                    # Draw eye landmarks specifically
                    for idx in LEFT_EYE + RIGHT_EYE:
                        pos = landmarks.landmark[idx]
                        px, py = int(pos.x * image_w), int(pos.y * image_h)
                        cv2.circle(display_frame, (px, py), 2, (0, 0, 255), -1)

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        detection_results["drowsy"] = True
                        cv2.putText(display_frame, "DROWSY ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    COUNTER = 0
                    detection_results["drowsy"] = False
                    
        # Update face markings status in detection results
        detection_results["show_face_markings"] = show_face_markings
    
    return display_frame

def camera_loop():
    global cap, frame_buffer, detection_active
    
    # Frame processing variables for optimization
    frame_count = 0
    process_every_n_frames = 2  # Only process every 2nd frame
    
    # Try to open the camera with multiple attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"Failed to open camera on attempt {attempt+1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    print("Could not open camera after multiple attempts")
                    detection_active = False
                    return
            
            # Set lower resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Reduce frame rate for better performance
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            break  # Camera opened successfully
        except Exception as e:
            print(f"Error opening camera: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)  # Wait before retrying
            else:
                print("Could not open camera after multiple attempts")
                detection_active = False
                return
    
    print("Camera initialized successfully")
    time.sleep(1)  # Allow camera to initialize fully
    
    # Create a default frame in case of errors
    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(default_frame, "Initializing camera...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', default_frame)
    frame_buffer = buffer.tobytes()
    
    frame_count = 0
    error_count = 0
    last_processed_frame = None
    
    while detection_active:
        try:
            ret, frame = cap.read()
            if not ret:
                error_count += 1
                print(f"Failed to read frame: {error_count}")
                if error_count > 5:  # If we get too many errors in a row
                    print("Too many frame read errors, resetting camera")
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    error_count = 0
                time.sleep(0.1)
                continue
                
            error_count = 0  # Reset error count on successful frame read
            frame_count += 1
            
            # Only process every nth frame to reduce CPU usage
            if frame_count % process_every_n_frames == 0:
                # Process the frame
                processed_frame = process_frame(frame)
                last_processed_frame = processed_frame
            else:
                # Use the last processed frame if available, otherwise process this frame
                if last_processed_frame is not None:
                    processed_frame = last_processed_frame
                else:
                    processed_frame = process_frame(frame)
                    last_processed_frame = processed_frame
            
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if ret:
                frame_buffer = buffer.tobytes()
            
            # Limit frame rate to reduce CPU usage
            time.sleep(0.06)  # ~15 FPS
            
        except Exception as e:
            print(f"Error in camera loop: {e}")
            time.sleep(0.1)
    
    # Release camera when done
    try:
        if cap is not None and cap.isOpened():
            cap.release()
            print("Camera released")
    except Exception as e:
        print(f"Error releasing camera: {e}")

def generate_frames():
    global frame_buffer
    
    # Create a default frame for when no frames are available
    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(default_frame, "Waiting for camera...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, default_buffer = cv2.imencode('.jpg', default_frame)
    default_frame_bytes = default_buffer.tobytes()
    
    # Send initial frame
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + default_frame_bytes + b'\r\n')
    
    while detection_active:
        try:
            if frame_buffer is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
            else:
                # If no frame is available, send the default frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + default_frame_bytes + b'\r\n')
            time.sleep(0.06)  # ~15 FPS
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            # Send an error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Stream error: {str(e)[:30]}", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, error_buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_buffer.tobytes() + b'\r\n')
            time.sleep(0.5)  # Longer delay on error

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active, camera_thread
    
    data = request.json
    new_state = data.get('active', False)
    
    if new_state and not detection_active:
        # Start detection
        detection_active = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.daemon = True
        camera_thread.start()
        return jsonify({"status": "started"})
    
    elif not new_state and detection_active:
        # Stop detection
        detection_active = False
        if camera_thread:
            camera_thread.join(timeout=1.0)
        return jsonify({"status": "stopped"})
    
    return jsonify({"status": "unchanged", "active": detection_active})

@app.route('/api/detection_status')
def detection_status():
    return jsonify({
        "active": detection_active,
        **detection_results
    })

@app.route('/api/toggle_face_markings', methods=['POST'])
def toggle_face_markings():
    global show_face_markings
    
    data = request.json
    show_face_markings = data.get('show', False)
    detection_results["show_face_markings"] = show_face_markings
    
    return jsonify({
        "status": "success",
        "show_face_markings": show_face_markings
    })

@app.route('/video_feed')
def video_feed():
    global detection_active, camera_thread
    
    # If detection is not active, start it
    if not detection_active:
        detection_active = True
        if camera_thread is None or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=camera_loop)
            camera_thread.daemon = True
            camera_thread.start()
            print("Camera thread started from video_feed route")
    
    # Return the video stream
    response = Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Add necessary CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
    response.headers.add('Pragma', 'no-cache')
    response.headers.add('Expires', '0')
    
    return response

if __name__ == '__main__':
    app.run(debug=True, threaded=True)