"""
Deepfake Detection using Hugging Face Model
============================================
Uses the open-deepfake-detection model from Hugging Face for accurate detection.

Model: prithivMLmods/open-deepfake-detection
Based on: google/vit-base-patch16-224

This provides more accurate detection than traditional CV methods by using
a Vision Transformer (ViT) trained specifically on deepfake datasets.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image

# Check for transformers availability
try:
    from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[!] Hugging Face transformers not installed!")
    print("    Install with: pip install transformers torch torchvision")


class HuggingFaceDeepfakeDetector:
    """
    Deepfake detector using Hugging Face's open-deepfake-detection model.

    This model is a fine-tuned Vision Transformer (ViT) that classifies
    images as 'Real' or 'Fake' with high accuracy.

    Features:
    - Deep learning-based detection (more accurate than CV methods)
    - Pre-trained on deepfake datasets
    - Multi-face support with per-face analysis
    - GPU acceleration if available
    """

    MODEL_ID = "prithivMLmods/open-deepfake-detection"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Hugging Face deepfake detector.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face transformers not installed. "
                "Install with: pip install transformers torch torchvision"
            )

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model: {self.MODEL_ID}")
        print(f"Device: {self.device}")

        # Load model and processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()

            # Also create a pipeline for simpler inference
            self.classifier = pipeline(
                "image-classification",
                model=self.MODEL_ID,
                device=0 if self.device == "cuda" else -1
            )

            print("[OK] Model loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Initialize face detector
        self._init_face_detector()

    def _init_face_detector(self):
        """Initialize OpenCV face detector."""
        self.face_detector = None
        model_path = os.path.join(os.path.dirname(__file__), 'face_detection_yunet.onnx')

        if os.path.exists(model_path):
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    model_path, "", (320, 320), 0.9, 0.3, 5000
                )
            except:
                pass

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image."""
        h, w = image.shape[:2]
        faces = []

        if self.face_detector is not None:
            self.face_detector.setInputSize((w, h))
            _, detections = self.face_detector.detect(image)
            if detections is not None:
                for det in detections:
                    x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    x, y = max(0, x), max(0, y)
                    fw, fh = min(fw, w - x), min(fh, h - y)
                    if fw > 30 and fh > 30:
                        faces.append((x, y, fw, fh))
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
            )
            faces = list(detections)

        return faces

    def classify_image(self, image: Image.Image) -> Dict:
        """
        Classify a PIL image as real or fake.

        Args:
            image: PIL Image

        Returns:
            Dict with 'label', 'confidence', 'scores'
        """
        # Use the pipeline for classification
        results = self.classifier(image)

        # Parse results
        fake_score = 0.0
        real_score = 0.0

        for result in results:
            label = result['label'].lower()
            score = result['score']

            if 'fake' in label or 'deepfake' in label or 'ai' in label:
                fake_score = score
            elif 'real' in label or 'authentic' in label:
                real_score = score

        # If model only outputs one class, infer the other
        if fake_score == 0.0 and real_score > 0:
            fake_score = 1.0 - real_score
        elif real_score == 0.0 and fake_score > 0:
            real_score = 1.0 - fake_score

        # Determine label
        is_fake = fake_score > real_score

        return {
            'label': 'Fake' if is_fake else 'Real',
            'is_fake': is_fake,
            'fake_score': fake_score,
            'real_score': real_score,
            'confidence': max(fake_score, real_score),
            'raw_results': results
        }

    def analyze_face(self, face_image: np.ndarray) -> Dict:
        """
        Analyze a single face image.

        Args:
            face_image: BGR numpy array of cropped face

        Returns:
            Analysis results
        """
        # Convert BGR to RGB and then to PIL
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Classify
        result = self.classify_image(pil_image)

        return result

    def analyze_image_array(self, image: np.ndarray) -> Dict:
        """
        Analyze image for deepfakes with multi-face support.

        Args:
            image: BGR numpy array

        Returns:
            Analysis results
        """
        results = {
            "faces_detected": 0,
            "face_analyses": [],
            "overall_score": 0.0,
            "verdict": "Unknown",
            "confidence": 0.0,
            "has_mixed_faces": False,
            "comparison_note": "",
            "model": self.MODEL_ID
        }

        # Detect faces
        faces = self.detect_faces(image)
        results["faces_detected"] = len(faces)

        if len(faces) == 0:
            # No faces - analyze full image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            full_result = self.classify_image(pil_image)

            results["overall_score"] = round(full_result['fake_score'], 3)
            results["confidence"] = round(full_result['confidence'] * 100, 1)
            results["verdict"] = self._get_verdict(full_result['fake_score'])
            results["full_image_analysis"] = True

            return results

        face_data = []

        for i, (x, y, w, h) in enumerate(faces):
            # Extract face with padding
            pad = int(min(w, h) * 0.2)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # Analyze face
            analysis = self.analyze_face(face_img)

            face_data.append({
                "face_id": i + 1,
                "location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "deepfake_probability": round(analysis['fake_score'], 3),
                "real_probability": round(analysis['real_score'], 3),
                "confidence": round(analysis['confidence'] * 100, 1),
                "is_likely_ai": analysis['is_fake'],
                "label": analysis['label'],
                # Compatibility with CV detector
                "frequency_score": round(analysis['fake_score'], 3),
                "texture_score": round(analysis['fake_score'], 3),
                "ai_indicators": round(analysis['fake_score'], 3),
                "real_indicators": round(analysis['real_score'], 3),
            })

        if not face_data:
            results["verdict"] = "Could not analyze faces"
            return results

        results["face_analyses"] = face_data

        # Multi-face comparison
        if len(face_data) >= 2:
            scores = [f['deepfake_probability'] for f in face_data]
            score_range = max(scores) - min(scores)

            if score_range > 0.3:
                results["has_mixed_faces"] = True
                ai_faces = [f for f in face_data if f['is_likely_ai']]
                real_faces = [f for f in face_data if not f['is_likely_ai']]

                if ai_faces and real_faces:
                    ai_ids = [f['face_id'] for f in ai_faces]
                    results["comparison_note"] = f"Face(s) {ai_ids} detected as AI-generated"

                results["overall_score"] = round(max(scores), 3)
            else:
                results["overall_score"] = round(np.mean(scores), 3)
        else:
            results["overall_score"] = round(face_data[0]['deepfake_probability'], 3)

        # Determine verdict
        results["verdict"] = self._get_verdict(results["overall_score"])
        results["confidence"] = round(
            max(results["overall_score"], 1 - results["overall_score"]) * 100, 1
        )

        return results

    def _get_verdict(self, score: float) -> str:
        """Convert score to verdict string (simplified to REAL or FAKE)."""
        if score < 0.50:
            return "REAL"
        else:
            return "FAKE"

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image from file path."""
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}

        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        return self.analyze_image_array(image)

    def analyze_video(self, video_path: str, sample_rate: int = 30) -> Dict:
        """Analyze video by sampling frames."""
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}

        results = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames_analyzed": 0,
            "frame_results": [],
            "overall_score": 0.0,
            "verdict": "Unknown",
            "confidence": 0.0,
            "faces_detected": 0,
            "model": self.MODEL_ID
        }

        frame_scores = []
        frame_count = 0

        print(f"Analyzing video with {self.MODEL_ID}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % sample_rate == 0:
                analysis = self.analyze_image_array(frame)
                if "error" not in analysis and analysis.get("faces_detected", 0) > 0:
                    frame_scores.append(analysis["overall_score"])
                    results["frame_results"].append({
                        "frame": frame_count,
                        "score": analysis["overall_score"],
                        "faces": analysis["faces_detected"]
                    })
                    results["faces_detected"] = max(
                        results["faces_detected"],
                        analysis["faces_detected"]
                    )
                results["frames_analyzed"] += 1

                if results["frames_analyzed"] % 5 == 0:
                    print(f"  Analyzed {results['frames_analyzed']} frames...")

        cap.release()

        if frame_scores:
            results["overall_score"] = round(float(np.mean(frame_scores)), 3)
            results["verdict"] = self._get_verdict(results["overall_score"])
            results["confidence"] = round(
                max(results["overall_score"], 1 - results["overall_score"]) * 100, 1
            )
        else:
            results["verdict"] = "No faces detected"

        print("Video analysis complete!")
        return results

    def draw_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on image."""
        output = image.copy()

        for face in results.get("face_analyses", []):
            loc = face["location"]
            x, y, w, h = loc["x"], loc["y"], loc["width"], loc["height"]
            prob = face["deepfake_probability"]
            is_ai = face.get("is_likely_ai", prob > 0.5)

            # Color coding (simplified)
            if prob < 0.50:
                color = (0, 255, 0)  # Green = REAL
            else:
                color = (0, 0, 255)  # Red = FAKE

            # Draw box
            thickness = 3 if prob >= 0.50 else 2
            cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)

            # Label
            label = "FAKE" if prob >= 0.50 else "REAL"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y-28), (x + lw + 10, y), color, -1)
            cv2.putText(output, label, (x+5, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overall verdict
        verdict = results.get("verdict", "Unknown")
        mixed = results.get("has_mixed_faces", False)

        verdict_text = verdict
        if mixed:
            verdict_text += " - MIXED"

        (vw, vh), _ = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(output, (5, 5), (vw + 20, vh + 25), (0, 0, 0), -1)
        cv2.putText(output, verdict_text, (10, vh + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Model label
        model_text = "Model: HF/open-deepfake-detection"
        (mw, mh), _ = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        h_img = output.shape[0]
        cv2.rectangle(output, (5, h_img - mh - 15), (mw + 15, h_img - 5), (0, 0, 0), -1)
        cv2.putText(output, model_text, (10, h_img - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return output


def main():
    """CLI demo."""
    print("=" * 60)
    print("  HUGGING FACE DEEPFAKE DETECTOR")
    print("  Model: prithivMLmods/open-deepfake-detection")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\n[ERROR] Required packages not installed!")
        print("        Install with: pip install transformers torch torchvision")
        return

    try:
        detector = HuggingFaceDeepfakeDetector()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize: {e}")
        return

    print("\nEnter image/video path or 'quit':")

    while True:
        path = input("\n> ").strip()
        if path.lower() in ['quit', 'exit', 'q']:
            break
        if not path or not os.path.exists(path):
            print("File not found")
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            print("\nAnalyzing image...")
            results = detector.analyze_image(path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            results = detector.analyze_video(path)
        else:
            print("Unsupported format")
            continue

        print("\n" + "="*40)
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Faces: {results.get('faces_detected', 0)}")
            print(f"Score: {results.get('overall_score', 0):.3f}")
            print(f"Verdict: {results.get('verdict', 'Unknown')}")
            print(f"Confidence: {results.get('confidence', 0):.1f}%")

            if results.get('has_mixed_faces'):
                print(f"\n[!] MIXED: {results.get('comparison_note', '')}")

            for face in results.get('face_analyses', []):
                status = "[FAKE]" if face.get('is_likely_ai') else "[REAL]"
                print(f"\nFace {face['face_id']}: {status}")
                print(f"  Fake probability: {face['deepfake_probability']*100:.1f}%")
                print(f"  Real probability: {face['real_probability']*100:.1f}%")
        print("="*40)


if __name__ == "__main__":
    main()  