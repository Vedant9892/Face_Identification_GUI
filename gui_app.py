import tkinter as tk
from tkinter import messagebox, font as tkfont, filedialog, scrolledtext
import subprocess
import os
import sys
from config import TRAINED_MODEL_DIR


class FaceRecognitionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        self.geometry("700x680")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold")
        self.button_font = tkfont.Font(family='Helvetica', size=12)
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self,
            text="Face Recognition System",
            font=self.title_font,
            bg="#263942",
            fg="white",
            pady=20
        )
        title_label.pack(fill=tk.X)
        instructions = tk.Label(
            self,
            text="Train the model using images from FACE_IMAGES folder,\nthen start live or video recognition.",
            font=('Helvetica', 10),
            bg="#f0f0f0",
            fg="#333333"
        )
        instructions.pack(pady=30)
        button_frame = tk.Frame(self, bg="#f0f0f0")
        button_frame.pack(pady=20)
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
        image_btn.grid(row=2, column=0, padx=10, pady=10)
        check_quality_btn = tk.Button(
            button_frame,
            text="Check Image Quality",
            font=self.button_font,
            bg="#2196F3",
            fg="white",
            width=20,
            height=2,
            command=self.check_image_quality,
            cursor="hand2"
        )
        check_quality_btn.grid(row=2, column=1, padx=10, pady=10)
        diagnostic_btn = tk.Button(
            button_frame,
            text="Diagnostic Mode",
            font=self.button_font,
            bg="#607D8B",
            fg="white",
            width=20,
            height=2,
            command=self.start_diagnostic,
            cursor="hand2"
        )
        diagnostic_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        self.status_label = tk.Label(
            self,
            text="Ready",
            font=('Helvetica', 9),
            bg="#f0f0f0",
            fg="#666666"
        )
        self.status_label.pack(pady=20)
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
        self.status_label.config(text="Training in progress... Please wait.", fg="orange")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "train_faces.py")
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                messagebox.showinfo("Success", "Model trained successfully!\nYou can now start live recognition.")
                self.status_label.config(text="Model trained successfully!", fg="green")
            else:
                messagebox.showerror("Error", f"Training failed:\n{result.stderr}")
                self.status_label.config(text="Training failed. Check console for details.", fg="red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error occurred during training.", fg="red")

    def train_model_enhanced(self):
        self.status_label.config(text="Enhanced training in progress... (6x data augmentation)", fg="orange")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "train_faces_enhanced.py")
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                messagebox.showinfo("Success", "Enhanced training completed!\n6x more training data created.\nBetter accuracy with augmentation!")
                self.status_label.config(text="Enhanced training successful! (6x augmentation applied)", fg="green")
            else:
                messagebox.showerror("Error", f"Enhanced training failed:\n{result.stderr}")
                self.status_label.config(text="Enhanced training failed.", fg="red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error during enhanced training.", fg="red")

    def start_live_recognition(self):
        if not self._check_models_exist():
            return
        self.status_label.config(text="Starting live recognition...", fg="blue")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "live_recognition.py")
            subprocess.Popen(
                [python_exe, script_path],
                cwd=os.path.dirname(__file__)
            )
            self.status_label.config(text="Live recognition started! Press 'q' in the window to quit.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live recognition:\n{str(e)}")
            self.status_label.config(text="Failed to start live recognition.", fg="red")

    def start_video_recognition(self):
        if not self._check_models_exist():
            return
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All Files", "*.*")
            ]
        )
        if not video_path:
            return
        self.status_label.config(text=f"Processing video: {os.path.basename(video_path)}...", fg="blue")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "video_recognition.py")
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
        if not self._check_models_exist():
            return
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        if not image_path:
            return
        self.status_label.config(text=f"Processing image: {os.path.basename(image_path)}...", fg="blue")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "image_recognition.py")
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

    def check_image_quality(self):
        self.status_label.config(text="Running image quality check...", fg="blue")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "check_image_quality.py")
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            output = (result.stdout or "") + (result.stderr or "")
            win = tk.Toplevel(self)
            win.title("Image Quality Check Results")
            win.geometry("700x500")
            text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=('Consolas', 9))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, output if output.strip() else "No output.")
            text.config(state=tk.DISABLED)
            self.status_label.config(text="Image quality check completed.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run quality check:\n{str(e)}")
            self.status_label.config(text="Quality check failed.", fg="red")

    def start_diagnostic(self):
        if not self._check_models_exist():
            return
        self.status_label.config(text="Starting diagnostic mode...", fg="blue")
        self.update()
        try:
            python_exe = sys.executable
            script_path = os.path.join(os.path.dirname(__file__), "diagnostic_tool.py")
            subprocess.Popen(
                [python_exe, script_path],
                cwd=os.path.dirname(__file__)
            )
            self.status_label.config(text="Diagnostic mode started! Press 'q' in the window to quit.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start diagnostic:\n{str(e)}")
            self.status_label.config(text="Failed to start diagnostic.", fg="red")

    def _check_models_exist(self):
        if not os.path.exists(TRAINED_MODEL_DIR):
            messagebox.showwarning(
                "No Model Found",
                "Please train the model first by clicking 'Train Model'."
            )
            return False
        found_models = False
        for item in os.listdir(TRAINED_MODEL_DIR):
            person_path = os.path.join(TRAINED_MODEL_DIR, item)
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
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.destroy()


if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.mainloop()
