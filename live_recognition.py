import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime
import geocoder
import os
from tkinter import Tk, filedialog
from config import TRAINED_MODEL_DIR, RECOGNITION_THRESHOLD, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT
from face_utils import load_embeddings, find_best_match

#live

def save_screenshot(frame):
    try:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        default_filename = f"face_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if not filepath:
            print("Screenshot canceled by user.")
            return False
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"✓ Screenshot saved successfully: {filepath}")
            return True
        else:
            print(f"✗ Failed to save screenshot to: {filepath}")
            return False
    except Exception as e:
        print(f"✗ Error saving screenshot: {e}")
        return False


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
    print("Starting Live Recognition... Press 'q' to quit, 's' to save screenshot.")
    window_name = 'Live Face Recognition - Press Q to quit, S to save screenshot'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT)
    screenshot_saved = False
    screenshot_flash_counter = 0
    
    frame_count = 0
    process_every_n_frames = 3
    cached_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        process_this_frame = (frame_count % process_every_n_frames == 0)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if process_this_frame:
            cached_faces = []
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
                                color = (0, 255, 100)
                                shadow_color = (0, 180, 70)
                                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), shadow_color, 3)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label_bg_height = 30
                                cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), (0, 180, 70), -1)
                                cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), color, 2)
                                cv2.putText(frame, display_name, (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                                age_badge_width = 60
                                cv2.rectangle(frame, (x, y+h), (x+age_badge_width, y+h+25), (0, 180, 70), -1)
                                cv2.rectangle(frame, (x, y+h), (x+age_badge_width, y+h+25), color, 2)
                                cv2.putText(frame, f"Age:{age}", (x+3, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            else:
                                color = (255, 100, 100)
                                shadow_color = (180, 70, 70)
                                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), shadow_color, 3)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                label_bg_height = 30
                                cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), (180, 70, 70), -1)
                                cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), color, 2)
                                cv2.putText(frame, "Unknown", (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            cached_faces.append({
                                'box': (x, y, w, h),
                                'name': display_name if folder_name is not None else "Unknown",
                                'age': age if folder_name is not None else "N/A",
                                'is_known': folder_name is not None
                            })
                        except Exception:
                            continue
            except Exception:
                pass
        else:
            for face_data in cached_faces:
                x, y, w, h = face_data['box']
                display_name = face_data['name']
                age = face_data['age']
                is_known = face_data['is_known']
                
                if is_known:
                    color = (0, 255, 100)
                    shadow_color = (0, 180, 70)
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), shadow_color, 3)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label_bg_height = 30
                    cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), (0, 180, 70), -1)
                    cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), color, 2)
                    cv2.putText(frame, display_name, (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    age_badge_width = 60
                    cv2.rectangle(frame, (x, y+h), (x+age_badge_width, y+h+25), (0, 180, 70), -1)
                    cv2.rectangle(frame, (x, y+h), (x+age_badge_width, y+h+25), color, 2)
                    cv2.putText(frame, f"Age:{age}", (x+3, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    color = (255, 100, 100)
                    shadow_color = (180, 70, 70)
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), shadow_color, 3)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label_bg_height = 30
                    cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), (180, 70, 70), -1)
                    cv2.rectangle(frame, (x, y-label_bg_height), (x+w, y), color, 2)
                    cv2.putText(frame, "Unknown", (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
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
        
        top_bar_height = 40
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (frame_width, top_bar_height), (15, 15, 15), -1)
        cv2.addWeighted(top_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.line(frame, (0, top_bar_height-1), (frame_width, top_bar_height-1), (0, 255, 200), 2)
        
        icon_color = (0, 255, 200)
        cv2.circle(frame, (15, 20), 6, icon_color, -1)
        cv2.putText(frame, "LIVE", (28, 27), cv2.FONT_HERSHEY_DUPLEX, 0.5, icon_color, 1, cv2.LINE_AA)
        
        screenshot_text = "Press 'S' to Capture"
        text_size = cv2.getTextSize(screenshot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        text_x = frame_width - text_size[0] - 15
        cv2.putText(frame, screenshot_text, (text_x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)
        
        info_panel_width = 320
        info_panel_height = 75
        panel_x = frame_width - info_panel_width - 10
        panel_y = frame_height - info_panel_height - 10
        
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (panel_x, panel_y), (frame_width - 10, frame_height - 10), (20, 20, 20), -1)
        cv2.addWeighted(bottom_overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), (frame_width - 10, frame_height - 10), (0, 200, 255), 2)
        cv2.line(frame, (panel_x, panel_y+30), (frame_width - 10, panel_y+30), (40, 40, 40), 1)
        
        cv2.circle(frame, (panel_x + 15, panel_y + 15), 4, (100, 200, 255), -1)
        cv2.putText(frame, "GPS", (panel_x + 25, panel_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)
        gps_coords_only = gps_text.replace("GPS: ", "")
        cv2.putText(frame, gps_coords_only, (panel_x + 15, panel_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.circle(frame, (panel_x + 15, panel_y + 58), 4, (255, 150, 100), -1)
        cv2.putText(frame, "TIME", (panel_x + 25, panel_y + 63), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 150, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, current_datetime, (panel_x + 70, panel_y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        if screenshot_flash_counter > 0:
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay, (0, 0), (frame_width, frame_height), (255, 255, 255), -1)
            cv2.addWeighted(flash_overlay, 0.5, frame, 0.5, 0, frame)
            text = "Screenshot Saved!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
            text_x = (frame_width - text_size[0]) // 2
            text_y = (frame_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            screenshot_flash_counter -= 1
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            clean_frame = frame.copy()
            if save_screenshot(clean_frame):
                screenshot_saved = True
                screenshot_flash_counter = 10
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()
