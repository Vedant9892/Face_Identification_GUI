import os
import numpy as np
from config import TRAINED_MODEL_DIR, RECOGNITION_THRESHOLD


def load_embeddings():
    all_embeddings = []
    all_folder_names = []
    person_info = {}
    if not os.path.exists(TRAINED_MODEL_DIR):
        print(f"Error: Trained Model directory not found at {TRAINED_MODEL_DIR}")
        return None, None, None
    for person_folder in os.listdir(TRAINED_MODEL_DIR):
        person_path = os.path.join(TRAINED_MODEL_DIR, person_folder)
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
                print(f"Loaded {len(embeddings)} embeddings for '{display_name}' (Age: {age})")
            except Exception as e:
                print(f"Error loading {encodings_file}: {e}")
    if not all_embeddings:
        print("Error: No trained models found. Please run train_faces.py first.")
        return None, None, None
    print(f"Total loaded: {len(all_embeddings)} embeddings from {len(person_info)} people")
    return np.array(all_embeddings), np.array(all_folder_names), person_info


def find_best_match(embedding, known_embeddings, known_folder_names, threshold=None):
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    if known_embeddings.size == 0:
        return None, float('inf')
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_dist_idx = np.argmin(distances)
    min_dist = float(distances[min_dist_idx])
    folder_name = known_folder_names[min_dist_idx]
    if min_dist < threshold:
        return folder_name, min_dist
    return None, min_dist
