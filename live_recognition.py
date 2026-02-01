import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
import geocoder
from config import RECOGNITION_THRESHOLD, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT
from face_utils import load_embeddings, find_best_match


def recognize_faces():
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return
    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Starting Live Recognition... Press 'q' to quit.")
    window_name = 'Live Face Recognition - Drag corners to resize (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)
    while True:
        ret, frame = cap.read()
        if not ret:
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
                        folder_name, min_dist = find_best_match(
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
        try:
            if not hasattr(recognize_faces, 'gps_coords'):
                g = geocoder.ip('me')
                if g.ok and g.latlng:
                    recognize_faces.gps_coords = f"GPS: {g.latlng[0]:.6f}, {g.latlng[1]:.6f}"
                else:
                    recognize_faces.gps_coords = "GPS: N/A"
            gps_text = recognize_faces.gps_coords
        except Exception:
            gps_text = "GPS: N/A"
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        overlay_width = 280
        overlay_height = 50
        cv2.rectangle(overlay, (frame_width - overlay_width, frame_height - overlay_height),
                     (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        y_offset = frame_height - 32
        x_offset = frame_width - overlay_width + 10
        cv2.putText(frame, gps_text, (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, current_datetime, (x_offset, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()
