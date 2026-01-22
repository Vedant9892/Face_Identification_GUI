import tkinter as tk
from tkinter import messagebox, font as tkfont, filedialog
import subprocess
import os
import sys

class FaceRecognitionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Face Recognition System")
        self.geometry("700x600")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        
        # Title Font
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold")
        self.button_font = tkfont.Font(family='Helvetica', size=12)
        
        # Setup UI
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self, 
            text="Face Recognition System", 
            font=self.title_font,
            bg="#263942",
            fg="white",
            pady=20
        )
        title_label.pack(fill=tk.X)
        
        # Instructions
        instructions = tk.Label(
            self,
            text="Train the model using images from FACE_IMAGES folder,\nthen start live or video recognition.",
            font=('Helvetica', 10),
            bg="#f0f0f0",
            fg="#333333"
        )
        instructions.pack(pady=30)
        
        # Frame for buttons
        button_frame = tk.Frame(self, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # Train Button
        train_btn = tk.Button(
            button_frame,
            text="Train Model",
            font=self.button_font,
            bg="#263942",
            fg="white",
            width=20,
            height=2,
            command=self.train_model,
            cursor="hand2"
        )
        train_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Enhanced Train Button (NEW)
        enhanced_train_btn = tk.Button(
            button_frame,
            text="Enhanced Training",
            font=self.button_font,
            bg="#9C27B0",
            fg="white",
            width=20,
            height=2,
            command=self.train_model_enhanced,
            cursor="hand2"
        )
        enhanced_train_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Live Recognition Button
        live_btn = tk.Button(
            button_frame,
            text="Live Camera Recognition",
            font=self.button_font,
            bg="#263942",
            fg="white",
            width=20,
            height=2,
            command=self.start_live_recognition,
            cursor="hand2"
        )
        live_btn.grid(row=1, column=0, padx=10, pady=10)
        
        # Video Recognition Button (NEW)
        video_btn = tk.Button(
            button_frame,
            text="Upload Video & Detect",
            font=self.button_font,
            bg="#4CAF50",
            fg="white",
            width=20,
            height=2,
            command=self.start_video_recognition,
            cursor="hand2"
        )
        video_btn.grid(row=1, column=1, padx=10, pady=10)
        
        # Image Recognition Button (NEW)
        image_btn = tk.Button(
            button_frame,
            text="Upload Image & Detect",
            font=self.button_font,
            bg="#FF9800",
            fg="white",
            width=20,
            height=2,
            command=self.start_image_recognition,
            cursor="hand2"
        )
        image_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        # Status Label
        self.status_label = tk.Label(
            self,
            text="Ready",
            font=('Helvetica', 9),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.status_label.pack(pady=20)
        
        # Exit Button
        exit_btn = tk.Button(
            self,
            text="Exit",
            font=self.button_font,
            bg="#ffffff",
            fg="#263942",
            width=10,
            command=self.quit_app,
            cursor="hand2"
        )
        exit_btn.pack(pady=10)
        
    def train_model(self):
        """Run the training script"""
        self.status_label.config(text="Training in progress... Please wait.", fg="orange")
        self.update()
        
        try:
            # Get the Python executable and script path
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "train_faces.py")
            
            # Run the training script
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            if result.returncode == 0:
                # Success
                messagebox.showinfo("Success", "Model trained successfully!\nYou can now start live recognition.")
                self.status_label.config(text="Model trained successfully!", fg="green")
            else:
                # Error
                messagebox.showerror("Error", f"Training failed:\n{result.stderr}")
                self.status_label.config(text="Training failed. Check console for details.", fg="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error occurred during training.", fg="red")
    
    def train_model_enhanced(self):
        """Run the enhanced training script with data augmentation"""
        self.status_label.config(text="Enhanced training in progress... (6x data augmentation)", fg="orange")
        self.update()
        
        try:
            # Get the Python executable and script path
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "train_faces_enhanced.py")
            
            # Run the enhanced training script
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            if result.returncode == 0:
                # Success
                messagebox.showinfo("Success", "Enhanced training completed!\n6x more training data created.\nBetter accuracy with augmentation!")
                self.status_label.config(text="Enhanced training successful! (6x augmentation applied)", fg="green")
            else:
                # Error
                messagebox.showerror("Error", f"Enhanced training failed:\n{result.stderr}")
                self.status_label.config(text="Enhanced training failed.", fg="red")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error during enhanced training.", fg="red")
    
    def start_live_recognition(self):
        """Start the live recognition window"""
        if not self._check_models_exist():
            return
        
        self.status_label.config(text="Starting live recognition...", fg="blue")
        self.update()
        
        try:
            # Get the Python executable and script path
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "live_recognition.py")
            
            # Run the live recognition script (non-blocking)
            subprocess.Popen(
                [python_exe, script_path],
                cwd=os.path.dirname(__file__)
            )
            
            self.status_label.config(text="Live recognition started! Press 'q' in the window to quit.", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live recognition:\n{str(e)}")
            self.status_label.config(text="Failed to start live recognition.", fg="red")
    
    def start_video_recognition(self):
        """Upload video and run face detection"""
        if not self._check_models_exist():
            return
        
        # Open file dialog to select video
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All Files", "*.*")
            ]
        )
        
        if not video_path:
            # User cancelled
            return
        
        self.status_label.config(text=f"Processing video: {os.path.basename(video_path)}...", fg="blue")
        self.update()
        
        try:
            # Get the Python executable and script path
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "video_recognition.py")
            
            # Run the video recognition script (non-blocking)
            subprocess.Popen(
                [python_exe, script_path, video_path],
                cwd=os.path.dirname(__file__)
            )
            
            self.status_label.config(
                text=f"Video recognition started! Processing: {os.path.basename(video_path)}. Press 'q' to quit.",
                fg="green"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start video recognition:\n{str(e)}")
            self.status_label.config(text="Failed to start video recognition.", fg="red")
    
    def start_image_recognition(self):
        """Upload image and run face detection"""
        if not self._check_models_exist():
            return
        
        # Open file dialog to select image
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        
        if not image_path:
            # User cancelled
            return
        
        self.status_label.config(text=f"Processing image: {os.path.basename(image_path)}...", fg="blue")
        self.update()
        
        try:
            # Get the Python executable and script path
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "image_recognition.py")
            
            # Run the image recognition script (non-blocking)
            subprocess.Popen(
                [python_exe, script_path, image_path],
                cwd=os.path.dirname(__file__)
            )
            
            self.status_label.config(
                text=f"Image opened! {os.path.basename(image_path)}. Press 'q' to close.",
                fg="green"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            self.status_label.config(text="Failed to process image.", fg="red")
    
    def _check_models_exist(self):
        """Check if any trained models exist"""
        trained_model_dir = os.path.join(os.path.dirname(__file__), "Trained_Model")
        
        if not os.path.exists(trained_model_dir):
            messagebox.showwarning(
                "No Model Found", 
                "Please train the model first by clicking 'Train Model'."
            )
            return False
        
        found_models = False
        for item in os.listdir(trained_model_dir):
            person_path = os.path.join(trained_model_dir, item)
            if os.path.isdir(person_path):
                encodings_file = os.path.join(person_path, "encodings.npz")
                if os.path.exists(encodings_file):
                    found_models = True
                    break
        
        if not found_models:
            messagebox.showwarning(
                "No Model Found", 
                "Please train the model first by clicking 'Train Model'."
            )
            return False
        
        return True
    
    def quit_app(self):
        """Exit the application"""
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.destroy()

if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.mainloop()
