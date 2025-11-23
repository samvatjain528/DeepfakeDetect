"""
Deepfake Detection Module - Enhanced Version
=============================================
Advanced deepfake detection with multi-face comparison support.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class DeepfakeDetector:
    """
    Enhanced deepfake and AI-generated image detector.

    Features:
    - Multi-scale frequency analysis
    - Cross-channel noise correlation
    - Micro-texture analysis
    - Multi-face comparison (detect which face is AI in group photos)
    """

    def __init__(self):
        """Initialize the detector."""
        self.face_detector = None
        model_path = self._get_face_model_path()
        if model_path:
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    model_path, "", (320, 320), 0.9, 0.3, 5000
                )
            except Exception:
                self.face_detector = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.use_dnn = self.face_detector is not None
        print("[OK] Enhanced Deepfake Detector initialized!")
        print(f"  Face detector: {'DNN (YuNet)' if self.use_dnn else 'Haar Cascade'}")

    def _get_face_model_path(self) -> str:
        """Download face detection model if needed."""
        model_path = os.path.join(os.path.dirname(__file__), 'face_detection_yunet.onnx')
        if not os.path.exists(model_path):
            print("Downloading face detection model...")
            import urllib.request
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("[OK] Model downloaded!")
            except Exception as e:
                print(f"Download failed: {e}, using Haar Cascade")
                return ""
        return model_path

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect all faces in image."""
        h, w = image.shape[:2]
        faces = []

        if self.use_dnn and self.face_detector is not None:
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

    # ========== ENHANCED ANALYSIS METHODS ==========

    def analyze_multiscale_frequency(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Multi-scale FFT analysis - AI has characteristic frequency artifacts.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

        ai_scores = []
        real_scores = []

        for size in [256, 128, 64]:
            resized = cv2.resize(gray, (size, size)).astype(np.float32)

            # Apply Hanning window
            window = np.outer(np.hanning(size), np.hanning(size))
            windowed = resized * window

            # FFT
            fft = np.fft.fftshift(np.fft.fft2(windowed))
            magnitude = np.log1p(np.abs(fft))
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

            center = size // 2

            # Radial profile analysis
            y, x = np.ogrid[:size, :size]
            r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
            max_r = min(center - 1, size - center - 1)

            radial_profile = np.array([
                np.mean(magnitude[r == i]) if np.any(r == i) else 0
                for i in range(max_r)
            ])

            if len(radial_profile) > 10:
                # Natural images follow 1/f power law (linear in log-log)
                x_log = np.log(np.arange(1, len(radial_profile) + 1))
                y_log = np.log(radial_profile + 1e-8)

                valid = np.isfinite(x_log) & np.isfinite(y_log)
                if np.sum(valid) > 5:
                    slope, _ = np.polyfit(x_log[valid], y_log[valid], 1)
                    y_pred = slope * x_log[valid] + _
                    ss_res = np.sum((y_log[valid] - y_pred) ** 2)
                    ss_tot = np.sum((y_log[valid] - np.mean(y_log[valid])) ** 2)
                    r_squared = 1 - ss_res / (ss_tot + 1e-8)

                    # Natural: slope -1 to -2, high R²
                    natural_slope = 1.0 if -2.5 < slope < -0.5 else 0.2
                    real_scores.append(r_squared * 0.5 + natural_slope * 0.5)
                    ai_scores.append((1 - r_squared) * 0.5 + (1 - natural_slope) * 0.5)
                else:
                    ai_scores.append(0.5)
                    real_scores.append(0.5)

            # Grid artifact detection (AI upscaling artifacts)
            h_profile = magnitude[center, center+5:center+size//3]
            v_profile = magnitude[center+5:center+size//3, center]

            if len(h_profile) > 5 and len(v_profile) > 5:
                h_var = np.var(h_profile)
                v_var = np.var(v_profile)

                # High variance in cardinal directions = grid artifacts
                grid_artifact = min((h_var + v_var) * 10, 1.0)
                ai_scores.append(grid_artifact)
                real_scores.append(1 - grid_artifact)

        weights = [0.5, 0.3, 0.2]
        ai_score = sum(s * w for s, w in zip(ai_scores[:3], weights)) if ai_scores else 0.5
        real_score = sum(s * w for s, w in zip(real_scores[:3], weights)) if real_scores else 0.5

        return float(np.clip(ai_score, 0, 1)), float(np.clip(real_score, 0, 1))

    def analyze_channel_correlation(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Cross-channel noise correlation - real cameras produce correlated noise.
        """
        if len(face_img.shape) != 3:
            return 0.5, 0.5

        img = cv2.resize(face_img, (128, 128))

        # Extract noise from each channel
        noises = []
        for i in range(3):
            channel = img[:, :, i].astype(np.float32)
            blurred = cv2.GaussianBlur(channel, (7, 7), 0)
            noise = (channel - blurred).flatten()
            noises.append(noise)

        # Cross-channel correlations
        try:
            corr_rg = np.corrcoef(noises[0], noises[1])[0, 1]
            corr_rb = np.corrcoef(noises[0], noises[2])[0, 1]
            corr_gb = np.corrcoef(noises[1], noises[2])[0, 1]
        except:
            return 0.5, 0.5

        correlations = [c if np.isfinite(c) else 0 for c in [corr_rg, corr_rb, corr_gb]]
        avg_corr = np.mean(np.abs(correlations))

        # Real camera: moderate correlation (0.2-0.6)
        # AI: often too low (<0.1) or too high (>0.8)
        if 0.15 < avg_corr < 0.65:
            real_score = 0.9
            ai_score = 0.1
        elif avg_corr < 0.08:
            real_score = 0.2
            ai_score = 0.85
        elif avg_corr > 0.8:
            real_score = 0.3
            ai_score = 0.7
        else:
            real_score = 0.5
            ai_score = 0.5

        # Noise level consistency across channels
        noise_stds = [np.std(n) for n in noises]
        std_range = max(noise_stds) - min(noise_stds)

        if std_range < 1.5:
            real_score = min(real_score + 0.1, 1.0)
        else:
            ai_score = min(ai_score + 0.15, 1.0)

        return float(ai_score), float(real_score)

    def analyze_micro_texture(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Micro-texture analysis - AI lacks fine skin detail (pores, fine lines).
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (256, 256))

        # Multi-scale high-pass filtering
        details = []
        for blur_size in [3, 7, 15]:
            blurred = cv2.GaussianBlur(gray, (blur_size*2+1, blur_size*2+1), 0)
            high_pass = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
            details.append(np.std(high_pass))

        # Laplacian variance (sharpness/detail)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)

        # Local Binary Pattern-like texture
        # Calculate local contrast
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        local_contrast = cv2.filter2D(gray, cv2.CV_64F, kernel)
        contrast_var = np.var(local_contrast)

        # Gabor filter responses (texture at different orientations)
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 8.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_responses.append(np.var(filtered))

        gabor_var = np.var(gabor_responses)
        gabor_mean = np.mean(gabor_responses)

        # Scoring
        # Real skin: high detail variance, varied Gabor responses
        detail_score = np.mean(details)

        if detail_score > 12 and lap_var > 200:
            real_score = 0.9
            ai_score = 0.1
        elif detail_score > 8 and lap_var > 100:
            real_score = 0.7
            ai_score = 0.3
        elif detail_score < 4 or lap_var < 50:
            real_score = 0.15
            ai_score = 0.85
        else:
            real_score = 0.45
            ai_score = 0.55

        # Gabor texture richness
        if gabor_var > gabor_mean * 0.2:
            real_score = min(real_score + 0.1, 1.0)
        else:
            ai_score = min(ai_score + 0.1, 1.0)

        return float(ai_score), float(real_score)

    def analyze_color_statistics(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Color distribution analysis - AI has subtle color anomalies.
        """
        if len(face_img.shape) != 3:
            return 0.5, 0.5

        img = cv2.resize(face_img, (128, 128))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Color channel statistics
        l_var = np.var(lab[:, :, 0])
        a_var = np.var(lab[:, :, 1])
        b_var = np.var(lab[:, :, 2])

        # Hue distribution
        hue = hsv[:, :, 0].flatten()
        hue_hist, _ = np.histogram(hue, bins=36, density=True)
        hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-8))

        # Color diversity
        colors = img.reshape(-1, 3)
        # Quantize colors for counting
        quantized = (colors // 16) * 16
        unique_colors = len(np.unique(quantized, axis=0))
        color_diversity = unique_colors / (128 * 128)

        # Gradient smoothness in color
        grad_l = np.diff(lab[:, :, 0].astype(np.float32), axis=1)
        grad_smoothness = np.mean(np.abs(np.diff(grad_l, axis=1)))

        # AI often has: low color diversity, smooth gradients, uniform a/b channels
        if color_diversity > 0.4 and (a_var + b_var) > 100:
            real_score = 0.8
            ai_score = 0.2
        elif color_diversity < 0.15 or (a_var + b_var) < 30:
            real_score = 0.2
            ai_score = 0.8
        else:
            real_score = 0.5
            ai_score = 0.5

        # Unnatural smoothness
        if grad_smoothness < 1.5:
            ai_score = min(ai_score + 0.2, 1.0)

        return float(ai_score), float(real_score)

    def analyze_noise_patterns(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Noise pattern analysis - AI has uniform/artificial noise.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (128, 128)).astype(np.float32)

        # Extract noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        noise_std = np.std(noise)
        noise_var = np.var(np.abs(noise))

        # Noise spatial distribution
        # Divide into quadrants and compare
        h, w = noise.shape
        quadrants = [
            noise[:h//2, :w//2],
            noise[:h//2, w//2:],
            noise[h//2:, :w//2],
            noise[h//2:, w//2:]
        ]
        quad_stds = [np.std(q) for q in quadrants]
        quad_var = np.var(quad_stds)

        # Real noise: varies spatially, std typically 3-15
        # AI noise: uniform, often too low or too high

        if 3 < noise_std < 15 and quad_var > 0.5:
            real_score = 0.85
            ai_score = 0.15
        elif noise_std < 2:
            real_score = 0.1
            ai_score = 0.9
        elif quad_var < 0.2:
            real_score = 0.25
            ai_score = 0.75
        else:
            real_score = 0.5
            ai_score = 0.5

        return float(ai_score), float(real_score)

    def analyze_edge_quality(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Edge analysis - AI edges are often too sharp or have ringing.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (256, 256))

        # Multi-threshold edge detection
        edges_low = cv2.Canny(gray, 20, 50)
        edges_mid = cv2.Canny(gray, 50, 100)
        edges_high = cv2.Canny(gray, 100, 200)

        density_low = np.mean(edges_low > 0)
        density_mid = np.mean(edges_mid > 0)
        density_high = np.mean(edges_high > 0)

        # Natural edge density ratio
        if density_low > 0:
            ratio_mid = density_mid / density_low
            ratio_high = density_high / (density_mid + 1e-8)
        else:
            ratio_mid, ratio_high = 0.5, 0.5

        # Gradient magnitude distribution
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)

        grad_hist, _ = np.histogram(gradient.flatten(), bins=50, density=True)
        grad_entropy = -np.sum(grad_hist * np.log(grad_hist + 1e-8))

        # Natural images: smooth density falloff, high gradient entropy
        natural_ratio = 0.3 < ratio_mid < 0.7 and 0.2 < ratio_high < 0.6

        if natural_ratio and grad_entropy > 3.5:
            real_score = 0.8
            ai_score = 0.2
        elif grad_entropy < 2.5:
            real_score = 0.25
            ai_score = 0.75
        else:
            real_score = 0.5
            ai_score = 0.5

        return float(ai_score), float(real_score)

    def analyze_compression_artifacts(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        JPEG artifact analysis - real photos have compression artifacts.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (128, 128))

        # Check 8x8 block boundaries
        block_diffs = []
        for i in range(8, 120, 8):
            row_diff = np.mean(np.abs(gray[i, :].astype(float) - gray[i-1, :].astype(float)))
            col_diff = np.mean(np.abs(gray[:, i].astype(float) - gray[:, i-1].astype(float)))
            block_diffs.extend([row_diff, col_diff])

        if block_diffs:
            artifact_strength = np.mean(block_diffs)
            artifact_var = np.var(block_diffs)
        else:
            artifact_strength, artifact_var = 0, 0

        # Real JPEGs: moderate artifacts (2-12) with variance
        # AI/PNG: very low or no artifacts
        if 2 < artifact_strength < 15 and artifact_var > 1:
            real_score = 0.75
            ai_score = 0.25
        elif artifact_strength < 1.5:
            real_score = 0.3
            ai_score = 0.7
        else:
            real_score = 0.5
            ai_score = 0.5

        return float(ai_score), float(real_score)

    def analyze_symmetry(self, face_img: np.ndarray) -> Tuple[float, float]:
        """
        Symmetry analysis - AI faces are often too perfect.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        gray = cv2.resize(gray, (128, 128))

        flipped = cv2.flip(gray, 1)
        diff = np.abs(gray.astype(float) - flipped.astype(float))
        symmetry = 1.0 - np.mean(diff) / 255.0

        # Natural faces: 0.70-0.88 symmetry
        # AI faces: often >0.90
        if 0.68 < symmetry < 0.86:
            real_score = 0.9
            ai_score = 0.1
        elif symmetry > 0.92:
            real_score = 0.15
            ai_score = 0.85
        elif symmetry > 0.88:
            real_score = 0.4
            ai_score = 0.6
        else:
            real_score = 0.5
            ai_score = 0.5

        return float(ai_score), float(real_score)

    # ========== MAIN ANALYSIS ==========

    def analyze_single_face(self, face_img: np.ndarray) -> Dict:
        """Analyze a single face region."""
        if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return {"error": "Face too small"}

        # Run all analyses
        analyses = {
            'frequency': self.analyze_multiscale_frequency(face_img),
            'channel_corr': self.analyze_channel_correlation(face_img),
            'micro_texture': self.analyze_micro_texture(face_img),
            'color': self.analyze_color_statistics(face_img),
            'noise': self.analyze_noise_patterns(face_img),
            'edges': self.analyze_edge_quality(face_img),
            'compression': self.analyze_compression_artifacts(face_img),
            'symmetry': self.analyze_symmetry(face_img),
        }

        # Weighted combination
        weights = {
            'frequency': 0.15,
            'channel_corr': 0.15,
            'micro_texture': 0.18,
            'color': 0.12,
            'noise': 0.12,
            'edges': 0.10,
            'compression': 0.08,
            'symmetry': 0.10,
        }

        total_ai = sum(analyses[k][0] * weights[k] for k in weights)
        total_real = sum(analyses[k][1] * weights[k] for k in weights)

        # Count strong indicators
        ai_strong = sum(1 for k in analyses if analyses[k][0] > 0.7)
        real_strong = sum(1 for k in analyses if analyses[k][1] > 0.7)

        # Calculate final score
        if real_strong >= 5:
            score = total_ai * 0.35
        elif ai_strong >= 5 and real_strong <= 2:
            score = min(total_ai * 1.5, 0.95)
        elif real_strong > ai_strong + 2:
            score = total_ai * 0.5
        elif ai_strong > real_strong + 2:
            score = min(total_ai * 1.3, 0.92)
        else:
            # Use differential scoring
            score = (total_ai * 0.6 + (1 - total_real) * 0.4)

        score = float(np.clip(score, 0, 1))

        return {
            'ai_score': round(total_ai, 3),
            'real_score': round(total_real, 3),
            'ai_strong': ai_strong,
            'real_strong': real_strong,
            'final_score': round(score, 3),
            'details': {k: {'ai': round(v[0], 3), 'real': round(v[1], 3)} for k, v in analyses.items()}
        }

    def analyze_image_array(self, image: np.ndarray) -> Dict:
        """Analyze image with multi-face comparison support."""
        results = {
            "faces_detected": 0,
            "face_analyses": [],
            "overall_score": 0.0,
            "verdict": "Unknown",
            "confidence": 0.0,
            "has_mixed_faces": False,
            "comparison_note": ""
        }

        faces = self.detect_faces(image)
        results["faces_detected"] = len(faces)

        if len(faces) == 0:
            results["verdict"] = "No faces detected"
            return results

        face_data = []

        for i, (x, y, w, h) in enumerate(faces):
            # Extract face with padding
            pad = int(min(w, h) * 0.15)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2 = min(image.shape[1], x + w + pad)
            y2 = min(image.shape[0], y + h + pad)
            face_img = image[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            analysis = self.analyze_single_face(face_img)

            if "error" in analysis:
                continue

            face_data.append({
                "face_id": i + 1,
                "location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "frequency_score": analysis['details']['frequency']['ai'],
                "texture_score": analysis['details']['micro_texture']['ai'],
                "color_score": analysis['details']['color']['ai'],
                "noise_score": analysis['details']['noise']['ai'],
                "consistency_score": analysis['details']['channel_corr']['ai'],
                "quality_score": analysis['details']['edges']['ai'],
                "symmetry_score": analysis['details']['symmetry']['ai'],
                "ai_indicators": analysis['ai_score'],
                "real_indicators": analysis['real_score'],
                "deepfake_probability": analysis['final_score'],
                "is_likely_ai": analysis['final_score'] > 0.55
            })

        if not face_data:
            results["verdict"] = "Could not analyze faces"
            return results

        results["face_analyses"] = face_data

        # Multi-face comparison
        if len(face_data) >= 2:
            scores = [f['deepfake_probability'] for f in face_data]
            score_range = max(scores) - min(scores)

            # If there's significant difference between faces
            if score_range > 0.25:
                results["has_mixed_faces"] = True
                ai_faces = [f for f in face_data if f['is_likely_ai']]
                real_faces = [f for f in face_data if not f['is_likely_ai']]

                if ai_faces and real_faces:
                    ai_ids = [f['face_id'] for f in ai_faces]
                    results["comparison_note"] = f"Face(s) {ai_ids} appear AI-generated while others look real"

                # Overall score: highest suspicious score
                results["overall_score"] = round(max(scores), 3)
            else:
                # All faces similar - use average
                results["overall_score"] = round(np.mean(scores), 3)
        else:
            results["overall_score"] = round(face_data[0]['deepfake_probability'], 3)

        # Determine verdict
        score = results["overall_score"]

        if score < 0.25:
            results["verdict"] = "Likely REAL"
            results["confidence"] = round((1 - score) * 100, 1)
        elif score < 0.40:
            results["verdict"] = "Probably REAL"
            results["confidence"] = round((1 - score) * 100, 1)
        elif score < 0.55:
            results["verdict"] = "UNCERTAIN"
            results["confidence"] = 50.0
        elif score < 0.70:
            results["verdict"] = "Probably FAKE"
            results["confidence"] = round(score * 100, 1)
        else:
            results["verdict"] = "Likely FAKE"
            results["confidence"] = round(min(score * 100, 95), 1)

        return results

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
            "faces_detected": 0
        }

        frame_scores = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % sample_rate == 0:
                analysis = self.analyze_image_array(frame)
                if "error" not in analysis and analysis["faces_detected"] > 0:
                    frame_scores.append(analysis["overall_score"])
                    results["frame_results"].append({
                        "frame": frame_count,
                        "score": analysis["overall_score"],
                        "faces": analysis["faces_detected"]
                    })
                    results["faces_detected"] = max(results["faces_detected"], analysis["faces_detected"])
                results["frames_analyzed"] += 1

        cap.release()

        if frame_scores:
            results["overall_score"] = round(float(np.mean(frame_scores)), 3)
            score = results["overall_score"]

            if score < 0.25:
                results["verdict"] = "Likely REAL"
                results["confidence"] = round((1 - score) * 100, 1)
            elif score < 0.40:
                results["verdict"] = "Probably REAL"
                results["confidence"] = round((1 - score) * 100, 1)
            elif score < 0.55:
                results["verdict"] = "UNCERTAIN"
                results["confidence"] = 50.0
            elif score < 0.70:
                results["verdict"] = "Probably FAKE"
                results["confidence"] = round(score * 100, 1)
            else:
                results["verdict"] = "Likely FAKE"
                results["confidence"] = round(min(score * 100, 95), 1)
        else:
            results["verdict"] = "No faces detected"

        return results

    def draw_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on image."""
        output = image.copy()

        for face in results.get("face_analyses", []):
            loc = face["location"]
            x, y, w, h = loc["x"], loc["y"], loc["width"], loc["height"]
            prob = face["deepfake_probability"]
            is_ai = face.get("is_likely_ai", prob > 0.55)

            # Color coding
            if prob < 0.25:
                color = (0, 255, 0)  # Green
            elif prob < 0.40:
                color = (0, 200, 100)  # Light green
            elif prob < 0.55:
                color = (0, 255, 255)  # Yellow
            elif prob < 0.70:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Draw box
            thickness = 3 if is_ai else 2
            cv2.rectangle(output, (x, y), (x+w, y+h), color, thickness)

            # Label
            label = f"{'AI' if is_ai else 'Real'}: {prob*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y-28), (x + lw + 10, y), color, -1)
            cv2.putText(output, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overall verdict
        verdict = results.get("verdict", "Unknown")
        confidence = results.get("confidence", 0)
        mixed = results.get("has_mixed_faces", False)

        verdict_text = f"{verdict} ({confidence:.0f}%)"
        if mixed:
            verdict_text += " - MIXED"

        (vw, vh), _ = cv2.getTextSize(verdict_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(output, (5, 5), (vw + 20, vh + 25), (0, 0, 0), -1)
        cv2.putText(output, verdict_text, (10, vh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Comparison note if present
        note = results.get("comparison_note", "")
        if note:
            (nw, nh), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (5, vh + 30), (nw + 15, vh + nh + 45), (0, 0, 150), -1)
            cv2.putText(output, note, (10, vh + nh + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output


def main():
    """CLI demo."""
    print("=" * 60)
    print("  ENHANCED DEEPFAKE DETECTOR")
    print("=" * 60)

    detector = DeepfakeDetector()

    print("\nFeatures:")
    print("- Multi-scale frequency analysis")
    print("- Cross-channel noise correlation")
    print("- Micro-texture detection")
    print("- Multi-face comparison (detects AI face in group photos)")
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
            print(f"Confidence: {results.get('confidence', 0)}%")

            if results.get('has_mixed_faces'):
                print(f"\n[!] MIXED: {results.get('comparison_note', '')}")

            for face in results.get('face_analyses', []):
                status = "[AI]" if face.get('is_likely_ai') else "[Real]"
                print(f"\nFace {face['face_id']}: {status} ({face['deepfake_probability']*100:.1f}%)")
        print("="*40)


if __name__ == "__main__":
    main()
