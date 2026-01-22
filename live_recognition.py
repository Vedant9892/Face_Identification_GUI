import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
from datetime import datetime
import geocoder

def load_embeddings():
    """Load embeddings from all person folders in Trained Model/"""
    trained_model_dir = os.path.join(os.path.dirname(__file__), "Trained_Model")
    
    if not os.path.exists(trained_model_dir):
        print(f"Error: Trained Model directory not found at {trained_model_dir}")
        return None, None, None
    
    all_embeddings = []
    all_folder_names = []
    person_info = {}  # Maps folder_name to (display_name, age)
    
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
                
                # Get folder name (for matching)
                folder_name = str(data.get('folder_name', data.get('name', person_folder)))
                
                # Get display info
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

def recognize_faces():
    # Load stored embeddings and person info
    known_embeddings, known_folder_names, person_info = load_embeddings()
    if known_embeddings is None:
        return

    # Initialize FaceNet and MTCNN
    try:
        embedder = FaceNet()
        detector = MTCNN()
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    # Open Webcam
    # cv2.CAP_DSHOW is often needed on Windows to avoid "can't grab frame" errors
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Live Recognition... Press 'q' to quit.")

    # Create resizable window with proper flags
    window_name = 'Live Face Recognition - Drag corners to resize (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1280, 720)  # Set initial size

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            results = detector.detect_faces(rgb_frame)

            # Only process if faces are detected
            if results:
                for result in results:
                    x, y, w, h = result['box']
                    x, y = abs(x), abs(y) # Fix negative coordinates
                    
                    # Extract Face
                    face = rgb_frame[y:y+h, x:x+w]
                    
                    try:
                        # Resize for FaceNet
                        face = cv2.resize(face, (160, 160))
                        
                        # Get Embedding
                        face_pixels = np.expand_dims(face, axis=0)
                        embedding = embedder.embeddings(face_pixels)[0]
                        

                        distances = []
                        for known_emb in known_embeddings:
                            dist = np.linalg.norm(embedding - known_emb)
                            distances.append(dist)
                        
                        distances = np.array(distances)
                        min_dist_idx = np.argmin(distances)
                        min_dist = distances[min_dist_idx]

                        if min_dist < 0.85:  
                            folder_name = known_folder_names[min_dist_idx]
                            
                        
                            if folder_name in person_info:
                                display_name, age = person_info[folder_name]
                            else:
                                display_name = folder_name
                                age = "N/A"
                            
                            confidence = f"{min_dist:.2f}"
                            # User Request: Identified = Red
                            color = (0, 0, 255) 
                            
                            # Draw Box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Display Name and Age ABOVE the box
                            cv2.putText(frame, display_name, (x, y-30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
                        else:
                            # User Request: Unknown = Green
                            color = (0, 255, 0)
                            
                            # Draw Box
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 2)

                    except Exception as e:
                        
                        continue
        
        except Exception as e:
     
            pass

    
        try:
            if not hasattr(recognize_faces, 'gps_coords'):
                # Try to get GPS from IP geolocation (only once)
                g = geocoder.ip('me')
                if g.ok and g.latlng:
                    recognize_faces.gps_coords = f"GPS: {g.latlng[0]:.6f}, {g.latlng[1]:.6f}"
                else:
                    # Fallback to default coordinates if unable to retrieve
                    recognize_faces.gps_coords = "GPS: N/A"
            gps_text = recognize_faces.gps_coords
        except Exception as e:
            gps_text = "GPS: N/A"
        
        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get frame dimensions for positioning
        frame_height, frame_width = frame.shape[:2]
        
        # Create semi-transparent overlay for text background (bottom-right)
        overlay = frame.copy()
        overlay_width = 280
        overlay_height = 50
        cv2.rectangle(overlay, (frame_width - overlay_width, frame_height - overlay_height), 
                     (frame_width, frame_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display GPS and DateTime in bottom-right corner (smaller text)
        y_offset = frame_height - 32
        x_offset = frame_width - overlay_width + 10
        cv2.putText(frame, gps_text, (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, current_datetime, (x_offset, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        # Check if window was closed (X button clicked) or 'q' was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Detect if window was closed with X button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
