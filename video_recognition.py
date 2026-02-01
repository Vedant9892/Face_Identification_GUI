import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import sys
from config import RECOGNITION_THRESHOLD, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT
from face_utils import load_embeddings, find_best_match


def recognize_video(video_path):
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return
    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    print(f"Processing video: {os.path.basename(video_path)}")
    print("Press 'q' to quit.")
    window_name = 'Video Face Recognition - Drag corners to resize (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = detector.detect_faces(rgb_frame)
            if results:
                for result in results:
                    x, y, w, h = result['box']
                    x, y = abs(x), abs(y)
                    face = rgb_frame[y:y+h, x:x+w]
                    try:
                        face = cv2.resize(face, (160, 160))
                        face_pixels = np.expand_dims(face, axis=0)
                        embedding = embedder.embeddings(face_pixels)[0]
                        folder_name, _ = find_best_match(
                            embedding, known_embeddings, known_folder_names, RECOGNITION_THRESHOLD
                        )
                        if folder_name is not None:
                            if folder_name in person_info:
                                display_name, age = person_info[folder_name]
                            else:
                                display_name = folder_name
                                age = "N/A"
                            color = (0, 0, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, display_name, (x, y-30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
                        else:
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
                    except Exception:
                        continue
        except Exception:
            pass
        cv2.imshow(window_name, frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_recognition.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    recognize_video(video_path)
