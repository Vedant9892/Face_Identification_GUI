import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import sys

def load_embeddings():
    """Load embeddings from all person folders in Trained Model/"""
    trained_model_dir = os.path.join(os.path.dirname(__file__), "Trained_Model")
    
    if not os.path.exists(trained_model_dir):
        print(f"Error: Trained_Model directory not found at {trained_model_dir}")
        return None, None, None
    
    all_embeddings = []
    all_folder_names = []
    person_info = {}  
    
    # Scan each person folder
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
                
                # Get display information and new data
                display_name = str(data.get('name', folder_name))
                age = str(data.get('age', 'N/A'))
                
                # Store person info
                person_info[folder_name] = (display_name, age)
                
                # Add all embeddings for this person
                for embedding in embeddings:
                    all_embeddings.append(embedding)
                    all_folder_names.append(folder_name)
                    
                print(f"Loaded {len(embeddings)} embeddings for '{display_name}' (Age: {age})")
            except Exception as e:
                print(f"Error loading {encodings_file}: {e}")
    
    if not all_embeddings:
        print("Error: No trained models found. Please run train_faces.py first.")
        return None, None, None
    
    print(f"Total loaded: {len(all_embeddings)} embeddings from {len(person_info)} people")
    return np.array(all_embeddings), np.array(all_folder_names), person_info

def recognize_video(video_path):
    """Process video file for face recognition"""
   
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return


    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # Open Video File
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    print(f"Processing video: {os.path.basename(video_path)}")
    print("Press 'q' to quit.")
    
    # Create resizable window with proper flags
    window_name = 'Video Face Recognition - Drag corners to resize (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 720)  # Set initial size

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Convert to RGB for detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            results = detector.detect_faces(rgb_frame)
            if results:
                for result in results:
                    x, y, w, h = result['box']
                    x, y = abs(x), abs(y) 
                    
                    # Extract Face
                    face = rgb_frame[y:y+h, x:x+w]
                    
                    try:
                        # Resize for FaceNet
                        face = cv2.resize(face, (160, 160))
                        
                        # Get Embedding
                        face_pixels = np.expand_dims(face, axis=0)
                        embedding = embedder.embeddings(face_pixels)[0]
                        
                        # Compare with known embeddings
                        distances = []
                        for known_emb in known_embeddings:
                            dist = np.linalg.norm(embedding - known_emb)
                            distances.append(dist)
                        
                        distances = np.array(distances)
                        min_dist_idx = np.argmin(distances)
                        min_dist = distances[min_dist_idx]
                        
                        # Recognition threshold
                        if min_dist < 0.85:  
                            folder_name = known_folder_names[min_dist_idx]
                            
                            # Get display name and age
                            if folder_name in person_info:
                                display_name, age = person_info[folder_name]
                            else:
                                display_name = folder_name
                                age = "N/A"
                            
                            # Recognized = Red
                            color = (0, 0, 255) 
                            
                            # Draw Box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Display Name and Age ABOVE the box
                            cv2.putText(frame, display_name, (x, y-30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
                        else:
                            # Unknown = Green
                            color = (0, 255, 0)
                            
                            # Draw Box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)

                    except Exception as e:
                        # Face extraction/resizing might fail
                        continue
        
        except Exception as e:
            # MTCNN detection might fail sometimes
            pass

        cv2.imshow(window_name, frame)

        
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 30ms = ~33 FPS
            break
        
        # Detect if window was closed with X button
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
