"""
Deepfake Detection GUI
======================
A beginner-friendly graphical interface for the deepfake detector.

Features:
- Drag and drop support
- Real-time analysis progress
- Visual results with annotated images
- Support for images and videos

Usage:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
from detector import DeepfakeDetector


class DeepfakeDetectorGUI:
    """
    A graphical user interface for the Deepfake Detector.

    This provides an easy-to-use interface for beginners to:
    - Load images or videos
    - Run deepfake analysis
    - View results with visual annotations
    """

    def __init__(self, root):
        """
        Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Deepfake Detector - Beginner Friendly")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Initialize detector
        self.detector = None
        self.current_image = None
        self.current_results = None
        self.is_analyzing = False

        # Setup UI
        self.setup_ui()

        # Initialize detector in background
        self.init_detector_async()

    def setup_ui(self):
        """Create all UI components."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        self.create_header()

        # Content area (split into left and right panels)
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel - Image display
        self.create_image_panel()

        # Right panel - Results and controls
        self.create_results_panel()

        # Footer with status
        self.create_footer()

    def create_header(self):
        """Create the header section with title and buttons."""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(
            header_frame,
            text="🔍 Deepfake Detector",
            font=('Helvetica', 18, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # Buttons frame
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)

        # Load Image button
        self.load_img_btn = ttk.Button(
            btn_frame,
            text="📷 Load Image",
            command=self.load_image
        )
        self.load_img_btn.pack(side=tk.LEFT, padx=5)

        # Load Video button
        self.load_vid_btn = ttk.Button(
            btn_frame,
            text="🎬 Load Video",
            command=self.load_video
        )
        self.load_vid_btn.pack(side=tk.LEFT, padx=5)

        # Analyze button
        self.analyze_btn = ttk.Button(
            btn_frame,
            text="🔬 Analyze",
            command=self.analyze,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

    def create_image_panel(self):
        """Create the left panel for image display."""
        # Left frame
        left_frame = ttk.LabelFrame(
            self.content_frame,
            text="Image/Video Preview",
            padding="10"
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Canvas for image display
        self.canvas = tk.Canvas(
            left_frame,
            bg='#2d2d2d',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text
        self.canvas.create_text(
            250, 200,
            text="Load an image or video\nto get started",
            fill='#888888',
            font=('Helvetica', 14),
            justify=tk.CENTER,
            tags="placeholder"
        )

        # Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)

    def create_results_panel(self):
        """Create the right panel for results display."""
        # Right frame
        right_frame = ttk.LabelFrame(
            self.content_frame,
            text="Analysis Results",
            padding="10"
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.configure(width=300)
        right_frame.pack_propagate(False)

        # Verdict display
        verdict_frame = ttk.Frame(right_frame)
        verdict_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            verdict_frame,
            text="Verdict:",
            font=('Helvetica', 12, 'bold')
        ).pack(anchor=tk.W)

        self.verdict_label = ttk.Label(
            verdict_frame,
            text="No analysis yet",
            font=('Helvetica', 16, 'bold'),
            foreground='gray'
        )
        self.verdict_label.pack(anchor=tk.W, pady=5)

        # Confidence display
        confidence_frame = ttk.Frame(right_frame)
        confidence_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            confidence_frame,
            text="Confidence:",
            font=('Helvetica', 12, 'bold')
        ).pack(anchor=tk.W)

        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            length=250,
            mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=5)

        self.confidence_label = ttk.Label(
            confidence_frame,
            text="0%",
            font=('Helvetica', 12)
        )
        self.confidence_label.pack(anchor=tk.W)

        # Detailed scores
        scores_frame = ttk.LabelFrame(right_frame, text="Detailed Scores", padding="10")
        scores_frame.pack(fill=tk.X, pady=10)

        # Score labels
        self.score_labels = {}
        scores = [
            ("Frequency Analysis", "freq"),
            ("Face Consistency", "consistency"),
            ("Quality Analysis", "quality"),
            ("Faces Detected", "faces")
        ]

        for text, key in scores:
            frame = ttk.Frame(scores_frame)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{text}:").pack(side=tk.LEFT)
            self.score_labels[key] = ttk.Label(frame, text="-")
            self.score_labels[key].pack(side=tk.RIGHT)

        # Explanation
        explain_frame = ttk.LabelFrame(right_frame, text="What This Means", padding="10")
        explain_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.explanation_text = tk.Text(
            explain_frame,
            wrap=tk.WORD,
            font=('Helvetica', 10),
            height=8,
            bg='#f5f5f5',
            relief=tk.FLAT
        )
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        self.explanation_text.insert(tk.END, "Load an image or video and click 'Analyze' to see results.\n\n"
                                            "The detector looks for:\n"
                                            "• Frequency patterns (artifacts from AI generation)\n"
                                            "• Face boundary inconsistencies\n"
                                            "• Image quality anomalies")
        self.explanation_text.config(state=tk.DISABLED)

    def create_footer(self):
        """Create the footer with status bar."""
        footer_frame = ttk.Frame(self.main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))

        # Status label
        self.status_label = ttk.Label(
            footer_frame,
            text="Initializing detector...",
            font=('Helvetica', 10)
        )
        self.status_label.pack(side=tk.LEFT)

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            footer_frame,
            mode='indeterminate',
            length=200
        )

        # File info
        self.file_label = ttk.Label(
            footer_frame,
            text="",
            font=('Helvetica', 10)
        )
        self.file_label.pack(side=tk.RIGHT)

    def init_detector_async(self):
        """Initialize the detector in a background thread."""
        def init():
            try:
                self.detector = DeepfakeDetector()
                self.root.after(0, lambda: self.status_label.config(text="Ready - Load an image or video to analyze"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Error initializing: {str(e)}"))

        thread = threading.Thread(target=init, daemon=True)
        thread.start()

    def load_image(self):
        """Open file dialog to load an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.load_file(file_path, "image")

    def load_video(self):
        """Open file dialog to load a video."""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.load_file(file_path, "video")

    def load_file(self, file_path, file_type):
        """
        Load an image or video file.

        Args:
            file_path: Path to the file
            file_type: "image" or "video"
        """
        self.current_file = file_path
        self.current_file_type = file_type

        # Update file label
        filename = os.path.basename(file_path)
        self.file_label.config(text=f"Loaded: {filename}")

        if file_type == "image":
            # Load and display image
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.display_image(image)
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Image loaded - Click 'Analyze' to detect deepfakes")
            else:
                messagebox.showerror("Error", f"Could not load image: {file_path}")
        else:
            # Load first frame of video
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.current_image = frame
                self.display_image(frame)
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Video loaded - Click 'Analyze' to detect deepfakes")
            else:
                messagebox.showerror("Error", f"Could not load video: {file_path}")

    def display_image(self, image, results=None):
        """
        Display an image on the canvas.

        Args:
            image: OpenCV image (BGR format)
            results: Optional detection results to draw
        """
        # Clear placeholder
        self.canvas.delete("placeholder")

        # Draw results if available
        if results and self.detector:
            image = self.detector.draw_results(image, results)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10:  # Canvas not yet rendered
            canvas_width = 500
            canvas_height = 400

        # Calculate scaling to fit canvas
        img_height, img_width = image_rgb.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        scale = min(scale, 1.0)  # Don't upscale

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize image
        image_resized = cv2.resize(image_rgb, (new_width, new_height))

        # Convert to PIL and then to PhotoImage
        pil_image = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(pil_image)

        # Clear canvas and display
        self.canvas.delete("all")
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)

    def on_canvas_resize(self, event):
        """Handle canvas resize event."""
        if self.current_image is not None:
            self.display_image(self.current_image, self.current_results)

    def analyze(self):
        """Run deepfake analysis on the loaded file."""
        if self.is_analyzing:
            return

        if self.detector is None:
            messagebox.showwarning("Wait", "Detector is still initializing. Please wait.")
            return

        self.is_analyzing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        self.progress_bar.start(10)
        self.status_label.config(text="Analyzing...")

        def run_analysis():
            try:
                if self.current_file_type == "image":
                    results = self.detector.analyze_image(self.current_file)
                else:
                    results = self.detector.analyze_video(self.current_file, sample_rate=15)

                self.root.after(0, lambda: self.show_results(results))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            finally:
                self.root.after(0, self.analysis_complete)

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

    def analysis_complete(self):
        """Called when analysis is complete."""
        self.is_analyzing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.config(text="Analysis complete")

    def show_results(self, results):
        """
        Display analysis results.

        Args:
            results: Dictionary with analysis results
        """
        self.current_results = results

        if "error" in results:
            self.verdict_label.config(text=results["error"], foreground='red')
            return

        # Update verdict
        verdict = results.get("verdict", "Unknown")
        if "REAL" in verdict:
            color = '#28a745'  # Green
        elif "FAKE" in verdict:
            color = '#dc3545'  # Red
        else:
            color = '#ffc107'  # Yellow

        self.verdict_label.config(text=verdict, foreground=color)

        # Update confidence
        confidence = results.get("confidence", 0)
        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"{confidence}%")

        # Update detailed scores
        self.score_labels["faces"].config(text=str(results.get("faces_detected", 0)))

        if results.get("face_analyses"):
            face = results["face_analyses"][0]
            self.score_labels["freq"].config(text=f"{face['frequency_score']:.2f}")
            self.score_labels["consistency"].config(text=f"{face['consistency_score']:.2f}")
            self.score_labels["quality"].config(text=f"{face['quality_score']:.2f}")
        else:
            self.score_labels["freq"].config(text="-")
            self.score_labels["consistency"].config(text="-")
            self.score_labels["quality"].config(text="-")

        # Update explanation
        self.explanation_text.config(state=tk.NORMAL)
        self.explanation_text.delete(1.0, tk.END)

        explanation = self.generate_explanation(results)
        self.explanation_text.insert(tk.END, explanation)
        self.explanation_text.config(state=tk.DISABLED)

        # Update image display with annotations
        if self.current_image is not None:
            self.display_image(self.current_image, results)

    def generate_explanation(self, results):
        """
        Generate a human-readable explanation of the results.

        Args:
            results: Analysis results dictionary

        Returns:
            Explanation string
        """
        verdict = results.get("verdict", "Unknown")
        confidence = results.get("confidence", 0)
        faces = results.get("faces_detected", 0)

        if faces == 0:
            return "No faces were detected in this image. The detector needs visible faces to analyze for deepfakes."

        explanation = f"Analysis detected {faces} face(s).\n\n"

        if "REAL" in verdict:
            explanation += f"The image appears to be REAL with {confidence}% confidence.\n\n"
            explanation += "This means:\n"
            explanation += "• Frequency patterns look natural\n"
            explanation += "• Face boundaries appear consistent\n"
            explanation += "• No obvious manipulation artifacts\n"
        elif "FAKE" in verdict:
            explanation += f"The image may be a DEEPFAKE with {confidence}% confidence.\n\n"
            explanation += "Warning signs detected:\n"

            if results.get("face_analyses"):
                face = results["face_analyses"][0]
                if face["frequency_score"] > 0.4:
                    explanation += "• Unusual frequency patterns (AI generation artifacts)\n"
                if face["consistency_score"] > 0.4:
                    explanation += "• Face boundary inconsistencies\n"
                if face["quality_score"] > 0.4:
                    explanation += "• Quality anomalies detected\n"
        else:
            explanation += f"Results are inconclusive ({confidence}% confidence).\n\n"
            explanation += "The image shows some suspicious signs but isn't clearly fake."

        explanation += "\n⚠️ Note: This is a basic detector for educational purposes. "
        explanation += "For critical decisions, use professional forensic tools."

        return explanation


def main():
    """Launch the GUI application."""
    root = tk.Tk()

    # Set icon (if available)
    try:
        # You can add an icon file here
        pass
    except:
        pass

    # Create and run app
    app = DeepfakeDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
