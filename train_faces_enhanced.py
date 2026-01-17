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
    txt_files = [f for f in os.listdir(person_dir) if f.endswith('.txt')]
    
    if not txt_files:
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

def augment_image(image):
    """Apply data augmentation to increase training data
    
    Creates variations of the image to make training more robust
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Horizontal flip (mirror)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # Brightness adjustments
    # Brighten
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
    augmented_images.append(bright)
    
    # Darken
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
    augmented_images.append(dark)
    
    # Slight rotation (-5 to +5 degrees)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Rotate +5 degrees
    matrix_plus = cv2.getRotationMatrix2D(center, 5, 1.0)
    rotated_plus = cv2.warpAffine(image, matrix_plus, (width, height))
    augmented_images.append(rotated_plus)
    
    # Rotate -5 degrees
    matrix_minus = cv2.getRotationMatrix2D(center, -5, 1.0)
    rotated_minus = cv2.warpAffine(image, matrix_minus, (width, height))
    augmented_images.append(rotated_minus)
    
    return augmented_images

if __name__ == "__main__":
    IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FACE_IMAGES")
    TRAINED_MODEL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Trained_Model")
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: Image directory not found at {IMAGES_DIR}")
        exit(1)
        
    print(f"Scanning {IMAGES_DIR}...")
    print("\n" + "="*70)
    print("ENHANCED TRAINING MODE - With Data Augmentation")
    print("="*70)
    print("Each image will be augmented to create 6 variations:")
    print("  1. Original")
    print("  2. Horizontal flip")
    print("  3. Brightened")
    print("  4. Darkened")
    print("  5. Rotated +5°")
    print("  6. Rotated -5°")
    print("="*70 + "\n")
    
    # Initialize models once
    detector = MTCNN()
    embedder = FaceNet()
    
    total_saved = 0
    total_original_images = 0
    
    # Process each person's folder separately
    for person_name in os.listdir(IMAGES_DIR):
        person_dir = os.path.join(IMAGES_DIR, person_name)
        
        if not os.path.isdir(person_dir):
            continue
        
        # Read person info from .txt file
        display_name, age = read_person_info(person_dir, person_name)
        logging.info(f"Processing images for: {display_name} (Age: {age})")
        
        person_embeddings = []
        original_count = 0
        augmented_count = 0
        
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
                
                # Detect face in ORIGINAL image first
                results = detector.detect_faces(image_rgb)
                
                if not results:
                    logging.warning(f"No face detected in {filename}")
                    continue
                
                # Extract face from original
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                face_original = image_rgb[y1:y2, x1:x2]
                
                # Apply augmentation to the CROPPED FACE
                augmented_faces = augment_image(face_original)
                
                # Process each augmented face
                for aug_idx, face in enumerate(augmented_faces):
                    try:
                        # FaceNet requires 160x160 input
                        face_resized = cv2.resize(face, (160, 160))
                        
                        # Get embedding
                        face_pixels = np.expand_dims(face_resized, axis=0)
                        embedding = embedder.embeddings(face_pixels)[0]
                        
                        person_embeddings.append(embedding)
                        
                        if aug_idx == 0:
                            original_count += 1
                            total_original_images += 1
                        else:
                            augmented_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing augmented image {aug_idx} of {filename}: {e}")
                
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
            
            print(f"✓ Saved {len(person_embeddings)} embeddings for '{display_name}' (Age: {age})")
            print(f"  └─ {original_count} original images → {len(person_embeddings)} total embeddings (6x augmentation)")
            total_saved += len(person_embeddings)
        else:
            print(f"✗ No embeddings extracted for '{person_name}'")
    
    print(f"\n{'='*70}")
    print(f"ENHANCED TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Original images processed: {total_original_images}")
    print(f"Total embeddings saved: {total_saved}")
    print(f"Augmentation factor: {total_saved/total_original_images if total_original_images > 0 else 0:.1f}x")
    print(f"Trained models location: {TRAINED_MODEL_BASE}")
    print(f"{'='*70}")
    print("\nBENEFITS OF AUGMENTATION:")
    print("  ✓ More robust recognition in different lighting")
    print("  ✓ Better handling of head tilts and angles")
    print("  ✓ Improved accuracy with limited training images")
    print("  ✓ Reduced overfitting")
    print(f"{'='*70}")
