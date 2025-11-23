"""
Deepfake Detection Web Application
==================================
A Flask-based web interface for the deepfake detector.

Supports two detector backends:
1. CV Detector - Traditional computer vision methods (fast, no GPU needed)
2. HF Detector - Hugging Face AI model (more accurate, downloads ~350MB model)

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider
import os
import base64
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from detector import DeepfakeDetector

# Try to import Hugging Face detector
try:
    from detector_hf import HuggingFaceDeepfakeDetector, HF_AVAILABLE
except ImportError:
    HF_AVAILABLE = False
    HuggingFaceDeepfakeDetector = None


class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detectors
cv_detector = DeepfakeDetector()
hf_detector = None  # Lazy load on first use

def get_detector(detector_type='cv'):
    """Get the appropriate detector based on type."""
    global hf_detector

    if detector_type == 'hf':
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face detector not available. "
                "Install with: pip install transformers torch torchvision"
            )
        if hf_detector is None:
            print("Loading Hugging Face model (first time may take a while)...")
            hf_detector = HuggingFaceDeepfakeDetector()
        return hf_detector

    return cv_detector

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return ext in ALLOWED_VIDEO_EXTENSIONS


def image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', hf_available=HF_AVAILABLE)


@app.route('/api/status')
def api_status():
    """Return API status and available detectors."""
    return jsonify({
        'status': 'ok',
        'detectors': {
            'cv': {
                'available': True,
                'name': 'CV Detector',
                'description': 'Traditional computer vision (fast, no GPU needed)'
            },
            'hf': {
                'available': HF_AVAILABLE,
                'name': 'AI Detector',
                'description': 'Hugging Face model (more accurate, requires torch)'
            }
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded image or video for deepfakes.

    Accepts form data with:
    - file: The image/video file
    - detector: 'cv' (default) or 'hf' for Hugging Face model

    Returns JSON with analysis results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get detector type from form data
    detector_type = request.form.get('detector', 'cv')

    # Determine file type
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    is_image = ext in ALLOWED_IMAGE_EXTENSIONS
    is_video = ext in ALLOWED_VIDEO_EXTENSIONS

    if not is_image and not is_video:
        return jsonify({'error': f'Unsupported file format: {ext}'}), 400

    # Save file temporarily
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Get the appropriate detector
        detector = get_detector(detector_type)

        if is_image:
            # Analyze image
            results = detector.analyze_image(filepath)

            # Generate annotated image
            if 'error' not in results:
                image = cv2.imread(filepath)
                annotated = detector.draw_results(image, results)
                results['annotated_image'] = image_to_base64(annotated)
                results['original_image'] = image_to_base64(image)
        else:
            # Analyze video
            results = detector.analyze_video(filepath, sample_rate=15)

            # Get first frame for display
            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            cap.release()

            if ret:
                annotated = detector.draw_results(frame, results)
                results['annotated_image'] = image_to_base64(annotated)
                results['original_image'] = image_to_base64(frame)

        # Add detector type to results
        results['detector_used'] = detector_type

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """
    Analyze image from URL.

    Accepts JSON with:
    - url: The image URL
    - detector: 'cv' (default) or 'hf' for Hugging Face model
    """
    data = request.get_json()
    url = data.get('url', '')
    detector_type = data.get('detector', 'cv')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        import urllib.request

        # Download image
        with urllib.request.urlopen(url) as response:
            image_data = response.read()

        # Convert to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Could not decode image from URL'}), 400

        # Get the appropriate detector
        detector = get_detector(detector_type)

        # Analyze
        results = detector.analyze_image_array(image)

        # Generate annotated image
        if 'error' not in results:
            annotated = detector.draw_results(image, results)
            results['annotated_image'] = image_to_base64(annotated)
            results['original_image'] = image_to_base64(image)

        # Add detector type to results
        results['detector_used'] = detector_type

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DEEPFAKE DETECTOR WEB APP")
    print("="*50)
    print("\nAvailable detectors:")
    print("  - CV Detector: Always available (fast, no GPU)")
    if HF_AVAILABLE:
        print("  - AI Detector: Available (Hugging Face model)")
    else:
        print("  - AI Detector: Not installed")
        print("    Install with: pip install transformers torch torchvision")
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
