import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_person_info(person_dir, person_folder_name):
    """Read person info from .txt file in their folder"""
    # Look for a .txt file in the person's folder
    txt_files = [f for f in os.listdir(person_dir) if f.endswith('.txt')]
    
    if not txt_files:
        # No .txt file found, use folder name
        return person_folder_name, "N/A"
    
    txt_file = os.path.join(person_dir, txt_files[0])
    
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        name = "N/A"
        age = "N/A"
        
        for line in lines:
            line = line.strip()
            if line.startswith('NAME:'):
                name = line.split('NAME:')[1].strip()
            elif line.startswith('AGE:'):
                age = line.split('AGE:')[1].strip()
        
        return name, age
    except Exception as e:
        logging.error(f"Error reading info file {txt_file}: {e}")
        return person_folder_name, "N/A"

def get_face_embeddings(image_dir):
    # Initialize models
    detector = MTCNN()
    embedder = FaceNet()
    
    known_embeddings = []
    known_names = []
    
    # Walk through the directory
    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        
        if not os.path.isdir(person_dir):
            continue
            
        logging.info(f"Processing images for: {person_name}")
        
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, filename)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not read image: {image_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = detector.detect_faces(image_rgb)
                
                if not results:
                    logging.warning(f"No face detected in {filename}")
                    continue
                
                # Assuming the first face is the target
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                face = image_rgb[y1:y2, x1:x2]
                
                # FaceNet requires 160x160 input
                face = cv2.resize(face, (160, 160))
                
                # Get embedding
                # FaceNet expects a list of faces
                face_pixels = np.expand_dims(face, axis=0)
                embedding = embedder.embeddings(face_pixels)[0]
                
                known_embeddings.append(embedding)
                known_names.append(person_name)
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                
    return np.array(known_embeddings), np.array(known_names)

if __name__ == "__main__":
    # Base directory for images
    # Images are now in FACE_IMAGES subfolder within the project
    IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FACE_IMAGES")
    
    # Base directory for trained models
    TRAINED_MODEL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained Model")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Image directory not found at {IMAGES_DIR}")
        exit(1)
        
    print(f"Scanning {IMAGES_DIR}...")
    
    # Initialize models once
    detector = MTCNN()
    embedder = FaceNet()
    
    total_saved = 0
    
    # Process each person's folder separately
    for person_name in os.listdir(IMAGES_DIR):
        person_dir = os.path.join(IMAGES_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        # Read person info from .txt file
        display_name, age = read_person_info(person_dir, person_name)
        logging.info(f"Processing images for: {display_name} (Age: {age})")
        
        person_embeddings = []
        
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(person_dir, filename)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not read image: {image_path}")
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = detector.detect_faces(image_rgb)
                
                if not results:
                    logging.warning(f"No face detected in {filename}")
                    continue
                
                # Assuming the first face is the target
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                face = image_rgb[y1:y2, x1:x2]
                
                # FaceNet requires 160x160 input
                face = cv2.resize(face, (160, 160))
                
                # Get embedding
                face_pixels = np.expand_dims(face, axis=0)
                embedding = embedder.embeddings(face_pixels)[0]
                
                person_embeddings.append(embedding)
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
        
        # Save this person's embeddings to their own folder
        if person_embeddings:
            person_model_dir = os.path.join(TRAINED_MODEL_BASE, person_name)
            os.makedirs(person_model_dir, exist_ok=True)
            
            output_file = os.path.join(person_model_dir, "encodings.npz")
            
            # Save embeddings, folder name, display name, and age
            np.savez(output_file, 
                    embeddings=np.array(person_embeddings), 
                    folder_name=person_name,
                    name=display_name,
                    age=age)
            
            print(f"✓ Saved {len(person_embeddings)} embeddings for '{display_name}' (Age: {age}) to {output_file}")
            total_saved += len(person_embeddings)
        else:
            print(f"✗ No embeddings extracted for '{person_name}'")
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Total embeddings saved: {total_saved}")
    print(f"Trained models location: {TRAINED_MODEL_BASE}")
    print(f"{'='*50}")
