import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import sys
from config import RECOGNITION_THRESHOLD, WINDOW_INITIAL_HEIGHT
from face_utils import load_embeddings, find_best_match


def recognize_image(image_path):
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return
    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image file: {image_path}")
        return
    print(f"Processing image: {os.path.basename(image_path)}")
    print("Press 'q' to close the window.")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        results = detector.detect_faces(rgb_image)
        if not results:
            print("No faces detected in the image.")
            cv2.putText(image, "No faces detected", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        else:
            print(f"Found {len(results)} face(s) in the image")
            for idx, result in enumerate(results, 1):
                x, y, w, h = result['box']
                x, y = abs(x), abs(y)
                face = rgb_image[y:y+h, x:x+w]
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
                        print(f"  Face {idx}: {display_name} (Age: {age}) - Distance: {min_dist:.3f}")
                        color = (0, 0, 255)
                        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(image, display_name, (x, y-35), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
                        cv2.putText(image, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color, 2)
                    else:
                        print(f"  Face {idx}: Unknown - Distance: {min_dist:.3f}")
                        color = (0, 255, 0)
                        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
                except Exception as e:
                    print(f"  Face {idx}: Error processing - {e}")
                    continue
    except Exception as e:
        print(f"Error during face detection: {e}")
    window_name = 'Image Face Recognition - Drag corners to resize (Press Q to close)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    height, width = image.shape[:2]
    max_width, max_height = 1280, WINDOW_INITIAL_HEIGHT
    if width > max_width or height > max_height:
        scale = min(max_width/width, max_height/height)
        cv2.resizeWindow(window_name, int(width*scale), int(height*scale))
    else:
        cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_recognition.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    recognize_image(image_path)
