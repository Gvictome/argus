# ARGUS Boot-Up Guide

Complete guide for setting up AI detection models on Raspberry Pi 5.

## Phase 3: AI Detection Models

This document covers the AI model setup that was deferred during initial scaffolding.

---

## 1. Model Overview

| Model | Purpose | Source | Size | Inference Time |
|-------|---------|--------|------|----------------|
| MobileNet-SSD | Object detection | TensorFlow Hub | ~23MB | ~50ms (CPU) |
| YOLOv8n | Object detection | Ultralytics | ~6MB | ~80ms (CPU) |
| FaceNet | Face recognition | Hugging Face | ~95MB | ~100ms (CPU) |
| MediaPipe Face | Face detection | Google | ~5MB | ~30ms (CPU) |

**With AI Hat+ 2 (Hailo-8L):**
- Inference drops to 5-15ms per frame
- Enables real-time 30fps processing

---

## 2. Installation

### 2.1 Core Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python ML dependencies
pip install tensorflow-lite-runtime  # Lighter than full TensorFlow
pip install opencv-python-headless   # No GUI needed on Pi
pip install mediapipe                # Google's ML solutions
pip install ultralytics              # YOLOv8
```

### 2.2 TensorFlow Lite (Recommended for Pi)

```bash
# TFLite is much lighter than full TensorFlow
pip install tflite-runtime

# Or for GPU/NPU support
pip install tensorflow==2.15.0
```

### 2.3 AI Hat+ 2 Setup (Optional)

```bash
# Install Hailo runtime
sudo apt install hailo-driver hailo-runtime

# Verify installation
hailortcli scan

# Install Python bindings
pip install hailo-platform
```

---

## 3. Model Downloads

### 3.1 Create Models Directory

```bash
mkdir -p ~/argus/models
cd ~/argus/models
```

### 3.2 Download Pre-trained Models

```bash
# MobileNet-SSD (TFLite)
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

# YOLOv8 Nano
pip install ultralytics
yolo export model=yolov8n.pt format=tflite

# Face Detection (MediaPipe - auto-downloads)
# No manual download needed

# FaceNet Embeddings
# Option 1: From Hugging Face
pip install facenet-pytorch

# Option 2: Pre-converted TFLite
wget https://github.com/AnirudhMaiya/FaceNet-TFLite/raw/master/facenet.tflite
```

### 3.3 Hailo Model Zoo (for AI Hat+ 2)

```bash
# Clone Hailo Model Zoo
git clone https://github.com/hailo-ai/hailo_model_zoo.git

# Download optimized models
hailo_model_zoo download --model yolov8n
hailo_model_zoo download --model mobilenet_v2
```

---

## 4. Detection Service Implementation

### 4.1 Motion Detection (Frame Differencing)

```python
# src/detection/motion.py

import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None

    def detect(self, frame):
        """
        Detect motion using frame differencing

        Args:
            frame: BGR numpy array

        Returns:
            List of motion regions [(x, y, w, h), ...]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return []

        # Compute difference
        diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))

        self.prev_frame = gray
        return regions
```

### 4.2 Object Detection (TFLite)

```python
# src/detection/objects.py

import numpy as np
import tflite_runtime.interpreter as tflite

class ObjectDetector:
    COCO_LABELS = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        # ... (full 80 COCO classes)
    ]

    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]

    def detect(self, frame):
        """
        Detect objects in frame

        Returns:
            List of detections: [(label, confidence, x, y, w, h), ...]
        """
        # Preprocess
        input_data = cv2.resize(frame, tuple(self.input_shape))
        input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        # Filter and convert
        detections = []
        h, w = frame.shape[:2]

        for i, score in enumerate(scores):
            if score > self.threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                x = int(xmin * w)
                y = int(ymin * h)
                width = int((xmax - xmin) * w)
                height = int((ymax - ymin) * h)

                label = self.COCO_LABELS[int(classes[i])]
                detections.append((label, float(score), x, y, width, height))

        return detections
```

### 4.3 Face Recognition

```python
# src/detection/faces.py

import cv2
import numpy as np
from pathlib import Path

class FaceRecognizer:
    def __init__(self, embeddings_path=None, threshold=0.6):
        self.threshold = threshold
        self.known_faces = {}  # {name: embedding}

        # Load face detector (Haar or MediaPipe)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load embedding model (FaceNet TFLite)
        self.embedding_model = None  # Load your model here

        if embeddings_path:
            self.load_embeddings(embeddings_path)

    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def get_embedding(self, face_crop):
        """Get face embedding vector"""
        # Resize to model input size (160x160 for FaceNet)
        face = cv2.resize(face_crop, (160, 160))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        # Get embedding from model
        # embedding = self.embedding_model.predict(face)
        # return embedding[0]

        # Placeholder - replace with actual model
        return np.random.rand(128)

    def recognize(self, frame):
        """
        Detect and recognize faces

        Returns:
            List of (name, confidence, x, y, w, h) or ("unknown", 0, ...)
        """
        faces = self.detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            embedding = self.get_embedding(face_crop)

            # Compare with known faces
            best_match = None
            best_distance = float('inf')

            for name, known_embedding in self.known_faces.items():
                distance = np.linalg.norm(embedding - known_embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name

            if best_distance < self.threshold:
                confidence = 1 - (best_distance / self.threshold)
                results.append((best_match, confidence, x, y, w, h))
            else:
                results.append(("unknown", 0, x, y, w, h))

        return results

    def enroll_face(self, name, frame):
        """Enroll a new face"""
        faces = self.detect_faces(frame)
        if len(faces) != 1:
            return False, "Expected exactly one face"

        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        embedding = self.get_embedding(face_crop)

        self.known_faces[name] = embedding
        return True, f"Enrolled {name}"

    def save_embeddings(self, path):
        """Save known face embeddings"""
        np.savez(path, **self.known_faces)

    def load_embeddings(self, path):
        """Load known face embeddings"""
        data = np.load(path)
        self.known_faces = {name: data[name] for name in data.files}
```

---

## 5. Detection Pipeline

### 5.1 Full Pipeline Integration

```python
# src/detection/pipeline.py

from src.detection.motion import MotionDetector
from src.detection.objects import ObjectDetector
from src.detection.faces import FaceRecognizer

class DetectionPipeline:
    def __init__(self, config):
        self.motion = MotionDetector(
            threshold=config.get('motion_threshold', 25)
        )
        self.objects = ObjectDetector(
            model_path=config['object_model_path'],
            threshold=config.get('detection_threshold', 0.5)
        )
        self.faces = FaceRecognizer(
            embeddings_path=config.get('embeddings_path'),
            threshold=config.get('face_threshold', 0.6)
        )

    def process(self, frame):
        """
        Full detection pipeline

        Returns:
            {
                'motion': [(x, y, w, h), ...],
                'objects': [(label, conf, x, y, w, h), ...],
                'faces': [(name, conf, x, y, w, h), ...],
                'alerts': [...]
            }
        """
        results = {
            'motion': [],
            'objects': [],
            'faces': [],
            'alerts': []
        }

        # 1. Motion detection (always runs - fast)
        motion_regions = self.motion.detect(frame)
        results['motion'] = motion_regions

        # 2. If motion detected, run object detection
        if motion_regions:
            detections = self.objects.detect(frame)
            results['objects'] = detections

            # 3. If humans detected, run face recognition
            humans = [d for d in detections if d[0] == 'person']
            if humans:
                faces = self.faces.recognize(frame)
                results['faces'] = faces

                # Generate alerts for unknown faces
                unknown = [f for f in faces if f[0] == 'unknown']
                if unknown:
                    results['alerts'].append({
                        'type': 'unknown_person',
                        'count': len(unknown),
                        'regions': unknown
                    })

            # Check for specific objects
            animals = [d for d in detections if d[0] in ['cat', 'dog', 'bird']]
            if animals:
                results['alerts'].append({
                    'type': 'animal_detected',
                    'animals': animals
                })

        return results
```

---

## 6. Performance Optimization

### 6.1 CPU Optimization

```python
# Reduce frame size for detection
DETECTION_SIZE = (640, 480)  # Process at lower res

# Skip frames when idle
SKIP_FRAMES = 2  # Process every 3rd frame

# Use threading
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
```

### 6.2 AI Hat+ 2 Acceleration

```python
# Using Hailo runtime
from hailo_platform import HEF, VDevice, ConfigureParams

class HailoDetector:
    def __init__(self, hef_path):
        self.hef = HEF(hef_path)
        self.device = VDevice()
        self.network_group = self.device.configure(self.hef)

    def detect(self, frame):
        # Preprocess
        input_data = preprocess(frame)

        # Run on NPU
        with self.network_group.activate():
            results = self.network_group.run([input_data])

        return postprocess(results)
```

---

## 7. Testing

### 7.1 Test Motion Detection

```bash
python -c "
from src.detection.motion import MotionDetector
import cv2

cap = cv2.VideoCapture(0)
detector = MotionDetector()

while True:
    ret, frame = cap.read()
    regions = detector.detect(frame)
    print(f'Motion regions: {len(regions)}')
"
```

### 7.2 Test Object Detection

```bash
python -c "
from src.detection.objects import ObjectDetector

detector = ObjectDetector('models/detect.tflite')
# Load test image and run detection
"
```

---

## 8. Systemd Service

```ini
# /etc/systemd/system/argus-detection.service

[Unit]
Description=ARGUS Detection Service
After=network.target argus.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/argus
ExecStart=/home/pi/argus/venv/bin/python -m src.detection.service
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 9. Next Steps

1. Download and test each model individually
2. Benchmark inference times on your Pi 5
3. Tune thresholds based on your environment
4. Set up face enrollment workflow
5. Configure alerts and automations

---

## Troubleshooting

### Model Loading Fails
```bash
# Check TFLite installation
python -c "import tflite_runtime.interpreter as tflite; print('OK')"

# Check model file exists
ls -la models/
```

### Slow Inference
- Reduce input resolution
- Use quantized (int8) models
- Enable AI Hat+ 2 if available
- Process fewer frames per second

### Memory Issues
```bash
# Check memory usage
free -m

# Reduce model size or use streaming inference
```
