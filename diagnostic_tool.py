import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os

def load_embeddings():
    """Load embeddings and show diagnostic info"""
    trained_model_dir = os.path.join(os.path.dirname(__file__), "Trained_Model")
    
    if not os.path.exists(trained_model_dir):
        print(f"Error: Trained_Model directory not found")
        return None, None, None
    
    all_embeddings = []
    all_folder_names = []
    person_info = {}
    
    print("\n" + "="*60)
    print("DIAGNOSTIC: Loading Trained Models")
    print("="*60)
    
    for person_folder in os.listdir(trained_model_dir):
        person_path = os.path.join(trained_model_dir, person_folder)
        
        if not os.path.isdir(person_path):
            continue
        
        encodings_file = os.path.join(person_path, "encodings.npz")
        
        if os.path.exists(encodings_file):
            try:
                data = np.load(encodings_file)
                embeddings = data['embeddings']
                
                folder_name = str(data.get('folder_name', data.get('name', person_folder)))
                display_name = str(data.get('name', folder_name))
                age = str(data.get('age', 'N/A'))
                
                person_info[folder_name] = (display_name, age)
                
                for embedding in embeddings:
                    all_embeddings.append(embedding)
                    all_folder_names.append(folder_name)
                
                print(f"✓ {folder_name}: {len(embeddings)} embeddings - {display_name} (Age: {age})")
                
            except Exception as e:
                print(f"✗ Error loading {encodings_file}: {e}")
    
    if not all_embeddings:
        print("Error: No trained models found!")
        return None, None, None
    
    print("="*60)
    print(f"Total: {len(all_embeddings)} embeddings from {len(person_info)} people\n")
    
    return np.array(all_embeddings), np.array(all_folder_names), person_info

def diagnose_recognition():
    """Test recognition with diagnostic output"""
    # Load embeddings
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return

    # Initialize models
    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # Open camera
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
    print("- Distance threshold is 0.85")
    print("This is only for diagnostic purposes")
    print("Works only on live camera")
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
                        
                        # Calculate distances to all known people
                        person_distances = {}
                        for i, known_emb in enumerate(known_embeddings):
                            dist = np.linalg.norm(embedding - known_emb)
                            folder_name = known_folder_names[i]
                            
                            if folder_name not in person_distances:
                                person_distances[folder_name] = []
                            person_distances[folder_name].append(dist)
                        
                        # Get minimum distance for each person
                        print("\n" + "-"*60)
                        print("Face Detected - Distance Scores:")
                        print("-"*60)
                        
                        min_distances = []
                        for folder_name in sorted(person_distances.keys()):
                            min_dist = min(person_distances[folder_name])
                            display_name = person_info[folder_name][0] if folder_name in person_info else folder_name
                            min_distances.append((min_dist, folder_name, display_name))
                        
                        # Sort by distance
                        min_distances.sort()
                        
                        # Display top 5 matches
                        for i, (dist, folder_name, display_name) in enumerate(min_distances[:5], 1):
                            status = "✓ MATCH" if dist < 0.85 else "✗ No match"
                            print(f"{i}. {display_name:20s} - Distance: {dist:.3f} {status}")
                        
                        # Show best match on frame
                        best_dist, best_folder, best_name = min_distances[0]
                        
                        if best_dist < 0.85:
                            color = (0, 0, 255)  # Red
                            label = f"{best_name} ({best_dist:.2f})"
                        else:
                            color = (0, 255, 0)  # Green
                            label = f"Unknown ({best_dist:.2f})"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    except Exception as e:
                        continue
        
        except Exception as e:
            pass

        cv2.imshow('Diagnostic Mode - Press Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    diagnose_recognition()
