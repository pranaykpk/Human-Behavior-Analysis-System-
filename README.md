# Human Behavior Detection System

This application combines an HTML/JavaScript frontend with a Flask backend to provide real-time human behavior detection, including emotion recognition, facial expression analysis, and drowsiness detection.

## Features

- Real-time face detection using YOLO
- Emotion recognition (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Facial expression analysis (Smile, Yawn, Neutral)
- Drowsiness detection based on eye aspect ratio
- Toggle button to start/stop detection
- Face markings visualization toggle
- Modern UI with responsive design

## Project Structure

```
├── app.py                 # Flask backend
├── main3.py               # Original detection script
├── index.html             # HTML frontend
├── public/                # Public assets
│   ├── manifest.json      # Web app manifest
│   └── logo.svg           # App logo
└── requirements.txt       # Python dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Webcam/camera device

### Installation

1. Install Python dependencies:

```bash
pip install flask flask-cors opencv-python numpy mediapipe ultralytics tensorflow
```

2. Run the Flask backend:

```bash
python app.py
```

### Performance Optimization

If you experience video lag or slow performance:

1. Reduce resolution in app.py by modifying the camera capture settings
2. Lower the frame rate by increasing the sleep time in the camera loop
3. Disable face markings when not needed
4. Ensure you have sufficient hardware resources (CPU/GPU)
5. Close other resource-intensive applications

## Usage

1. Open your browser and navigate to http://localhost:5000
2. Use the toggle switch to start/stop the detection
3. View real-time detection results in the right panel
4. Toggle face markings to visualize facial landmarks


## Model Information

- Face detection: YOLOv8n-face model
- Emotion recognition: Custom trained CNN model
- Face landmarks: MediaPipe FaceMesh

## Notes

- The application requires camera access
- For best results, ensure good lighting conditions
- The drowsiness detection is based on eye aspect ratio and may need calibration for different users
