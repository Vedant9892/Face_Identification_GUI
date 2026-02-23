import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from config import RECOGNITION_THRESHOLD
from face_utils import load_embeddings, find_best_match

def diagnose_recognition():
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
    print("="*60)
    print("DIAGNOSTIC MODE - Testing Recognition")
    print("="*60)
    print("Instructions:")
    print("- Position face in camera")
    print("- You'll see distance scores for each person")
    print("- Lower distance = better match")
    print("- Press 'q' to quit")
    print("="*60 + "\n")
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
                        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
                        person_distances = {}
                        for i, folder_name in enumerate(known_folder_names):
                            if folder_name not in person_distances:
                                person_distances[folder_name] = []
                            person_distances[folder_name].append(float(distances[i]))
                        print("\n" + "-"*60)
                        print("Face Detected - Distance Scores:")
                        print("-"*60)
                        min_distances = []
                        for folder_name in sorted(person_distances.keys()):
                            min_dist = min(person_distances[folder_name])
                            display_name = person_info[folder_name][0] if folder_name in person_info else folder_name
                            min_distances.append((min_dist, folder_name, display_name))
                        min_distances.sort()
                        for i, (dist, folder_name, display_name) in enumerate(min_distances[:5], 1):
                            status = "✓ MATCH" if dist < RECOGNITION_THRESHOLD else "✗ No match"
                            print(f"{i}. {display_name:20s} - Distance: {dist:.3f} {status}")
                        best_dist, best_folder, best_name = min_distances[0]
                        if best_dist < RECOGNITION_THRESHOLD:
                            color = (0, 0, 255)
                            label = f"{best_name} ({best_dist:.2f})"
                        else:
                            color = (0, 255, 0)
                            label = f"Unknown ({best_dist:.2f})"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception:
                        continue
        except Exception:
            pass
        cv2.imshow('Diagnostic Mode - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#MAIN PUSH

if __name__ == "__main__":
    diagnose_recognition()
