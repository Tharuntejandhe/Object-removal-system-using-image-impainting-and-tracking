import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import sys
import os

class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Object Removal Pipeline")
        self.root.geometry("600x500")
        self.root.configure(padx=20, pady=20)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Helvetica", 12), padding=6)
        style.configure("TLabel", font=("Helvetica", 12))
        
        # Video Selection
        self.video_path = tk.StringVar()
        
        header = ttk.Label(root, text="Video Object Remover", font=("Helvetica", 18, "bold"))
        header.pack(pady=(0, 20))
        
        file_frame = ttk.Frame(root)
        file_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_frame, text="Input Video:").pack(side=tk.LEFT, padx=(0, 10))
        self.entry = ttk.Entry(file_frame, textvariable=self.video_path, width=40)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(file_frame, text="Browse", command=self.browse_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Pipeline Steps
        steps_frame = ttk.LabelFrame(root, text="Pipeline Steps", padding=15)
        steps_frame.pack(fill=tk.X, pady=15)
        
        self.btn_extract = ttk.Button(steps_frame, text="1. Extract Media", command=self.run_extraction)
        self.btn_extract.pack(fill=tk.X, pady=5)
        
        self.btn_select = ttk.Button(steps_frame, text="2. Select Object", command=self.run_selection)
        self.btn_select.pack(fill=tk.X, pady=5)
        
        self.btn_track = ttk.Button(steps_frame, text="3. Track Object (SAM 2)", command=self.run_tracking)
        self.btn_track.pack(fill=tk.X, pady=5)
        
        self.btn_inpaint = ttk.Button(steps_frame, text="4. Erase & Clean Up", command=self.run_inpainting)
        self.btn_inpaint.pack(fill=tk.X, pady=5)
        
        # Logs
        log_frame = ttk.LabelFrame(root, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.text_log = tk.Text(log_frame, height=10, state=tk.DISABLED, bg="#f0f0f0", font=("Courier", 11))
        self.text_log.pack(fill=tk.BOTH, expand=True)
        
    def browse_video(self):
        file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")])
        if file:
            self.video_path.set(file)
            
    def log(self, message):
        self.text_log.config(state=tk.NORMAL)
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
        self.text_log.config(state=tk.DISABLED)
        
    def run_subprocess(self, cmd, success_msg="Finished successfully."):
        def task():
            self.disable_buttons()
            self.log(f"Running: {' '.join(cmd)}")
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    self.root.after(0, self.log, line.strip())
                process.wait()
                
                if process.returncode == 0:
                    self.root.after(0, self.log, success_msg)
                else:
                    self.root.after(0, self.log, "ERROR: Process failed.")
            except Exception as e:
                self.root.after(0, self.log, f"ERROR: {e}")
            finally:
                self.root.after(0, self.enable_buttons)
                
        threading.Thread(target=task, daemon=True).start()

    def disable_buttons(self):
        for btn in [self.btn_extract, self.btn_select, self.btn_track, self.btn_inpaint]:
            btn.state(['disabled'])

    def enable_buttons(self):
        for btn in [self.btn_extract, self.btn_select, self.btn_track, self.btn_inpaint]:
            btn.state(['!disabled'])

    def run_extraction(self):
        video = self.video_path.get()
        if not video:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        cmd = ["conda", "run", "-n", "myenv", "python", "run_pipeline.py", "--video", video]
        self.run_subprocess(cmd, "Extraction complete. Now click 'Select Object'.")

    def run_selection(self):
        # Open in separate console to allow OpenCV drawing
        cmd = ["conda", "run", "-n", "myenv", "python", "scripts/select_object.py", "--frames_dir", "frames", "--mode", "box"]
        self.run_subprocess(cmd, "Selection saved. Now click 'Track Object'.")

    def run_tracking(self):
        cmd = ["conda", "run", "-n", "myenv", "python", "run_pipeline.py", "--track"]
        self.run_subprocess(cmd, "Tracking complete. Now click 'Erase & Clean Up'.")

    def run_inpainting(self):
        cmd = ["conda", "run", "-n", "myenv", "python", "run_pipeline.py", "--inpaint", "--cleanup"]
        self.run_subprocess(cmd, "Pipeline complete! Final video saved to output/final_video.mp4")

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
