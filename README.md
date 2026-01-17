# FaceDetection-CNN: Real-Time Face Recognition System

A powerful deep learning-based face recognition system built with **FaceNet** and **MTCNN**, featuring a user-friendly GUI and personalized information display.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

##  Features
**High-Accuracy Face Detection** - Uses MTCNN (Multi-task Cascaded Convolutional Networks)
-  **Deep Learning Recognition** - Powered by FaceNet CNN for 128D face embeddings
-  **Intuitive GUI** - Simple Tkinter interface for easy operation
-  **Personalized Display** - Shows names and ages from user-defined info files
- **Organized Storage** - Individual model files per person for scalability
- **Color-Coded Detection**:
   **Green Box** → Unknown/Unrecognized face
   **Red Box** → Recognized face with name and age displayed
- **One-Shot Learning** - Train once, recognize instantly
- **Easy Expansion** - Simply add new person folders to scale up

## Demo

When a person is detected, the system displays:
- **Red bounding box** around the face
- **Full name** above the box
- **Age** directly below the name
- Continuous real-time tracking

Unknown faces are shown with a **green box** and "Unknown" label.

## Requirements

- Python 3.11+
- Webcam
- Windows (tested) / Linux / macOS

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Vedant9892/Face_Identification_GUI
cd FaceDetection-CNN
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- tensorflow
- mtcnn
- opencv-python
- numpy
- keras-facenet
- scipy

## Usage

### Quick Start

1. **Run the GUI**:
   ```bash
   python gui_app.py
   ```

2. **Train the Model**:
   - Click the **"Train Model"** button
   - Wait for processing to complete

3. **Start Recognition**:
   - Click **"Start Live Recognition"**
   - Your webcam will open
   - Press **'q'** to quit

### Command Line Usage

**Training:**
```bash
python train_faces.py
```

**Live Recognition:**
```bash
python live_recognition.py
```

##  Project Structure

```
FaceDetection-CNN/
├── FACE_IMAGES/              # Training data
│   ├── person1/
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── person1.txt       # NAME and AGE info
│   ├── person2/
│   └── person3/
│
├── Trained Model/            # Saved models (auto-generated)
│   ├── person1/
│   │   └── encodings.npz
│   ├── person2/
│   └── person3/
│
├── gui_app.py               # Main GUI application
├── train_faces.py           # Training script
├── live_recognition.py      # Live recognition script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

##  Adding New People [IMPORTANT]

1. **Create a folder** in `FACE_IMAGES/`:
   ```
   FACE_IMAGES/person4/
   ```

2. **Add 2-5 clear face images** (JPG/PNG):
   ```
   person1/img1.jpg
   person1/img2.jpg
   ```

3. **Create info file** `person1.txt`:
   ```
   NAME:John Doe
   AGE: 25
   ```

4. **Re-train the model**:
   - Run GUI and click "Train Model"
   - OR run `python train_faces.py`

5. **Done!** The system will now recognize the new person.

## Technical Details

### Models Used

| Model | Purpose | Details |
|-------|---------|---------|
| **MTCNN** | Face Detection | Multi-task CNN for detecting faces and facial landmarks |
| **FaceNet** | Face Recognition | Generates 128-dimensional embeddings for face matching |

### Recognition Process

1. **Detection**: MTCNN detects faces in video frames
2. **Alignment**: Face regions are extracted and resized to 160x160
3. **Embedding**: FaceNet generates a 128D vector representing the face
4. **Matching**: Euclidean distance compared with stored embeddings
5. **Threshold**: Distance < 0.85 → Recognized, else Unknown

### Storage Format

Each person's data is stored in `Trained Model/personX/encodings.npz` containing:
- **embeddings**: 128D face vectors (NumPy array)
- **folder_name**: Folder identifier
- **name**: Display name from .txt file
- **age**: Age from .txt file

## Configuration

### Adjust Recognition Sensitivity

Edit `live_recognition.py`, line ~119:

```python
if min_dist < 0.85:  # Lower = stricter, Higher = more lenient
```

Recommended values:
- **0.5-0.6**: Very strict (high accuracy, may miss variations)
- **0.7-0.8**: Balanced (default)
- **0.85-1.0**: Lenient (recognizes with variations, may have false positives)

## Troubleshooting

### Camera Not Opening

**Issue**: `can't grab frame` error

**Solution**: Already fixed with `cv2.CAP_DSHOW`. If still occurring:
```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows
# cap = cv2.VideoCapture(0)  # Linux/Mac
```

### Module Not Found Errors

**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

### No Face Detected During Training

**Solution**: Ensure:
- Images have clear, well-lit faces
- Face is not too small or at extreme angles
- Images are in JPG/PNG format

##  Future Enhancements

- [ ] Multi-face tracking
- [ ] Face detection from video files
- [ ] Export recognition logs
- [ ] GPU acceleration support
- [ ] Mobile app version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For questions or support, please open an issue on GitHub.

---

**Made using Deep Learning and Computer Vision**
