# 🔍 Deepfake Detector

An AI-powered web application for detecting deepfake and synthetic media using multiple detection methods. Built with Python, Flask, and advanced computer vision techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 Multiple Detection Methods

1. **CV Detector** (Computer Vision)
   - Fast, no GPU required
   - 15 advanced analysis methods including:
     - Multi-scale frequency analysis
     - Cross-channel noise correlation
     - Micro-texture analysis
     - Eye, teeth, hair, and skin pore analysis
     - Lighting and blur consistency checks
     - Edge quality and compression artifact detection

2. **AI Detector** (HuggingFace)
   - Uses Vision Transformer (ViT) model
   - Pre-trained on deepfake datasets
   - Model: `prithivMLmods/open-deepfake-detection`
   - GPU acceleration support

3. **SVM Detector** (Machine Learning)
   - ViT embeddings + SVM classifier
   - Trainable on custom datasets
   - Best for specialized use cases
   - Supports model persistence

### 🖼️ Multi-Face Detection
- Analyzes multiple faces in group photos
- Detects which specific faces are AI-generated
- Provides per-face analysis and confidence scores

### 📹 Video Support
- Analyzes videos frame-by-frame
- Supports MP4, AVI, MOV formats
- Configurable frame sampling rate

### 🌐 Modern Web Interface
- Drag & drop file upload
- URL-based image analysis
- Real-time results with visual feedback
- Responsive design for all devices
- Beautiful gradient UI with SVG icons

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DeepfakeDetect.git
   cd DeepfakeDetect
   ```

2. **Install base dependencies**
   ```bash
   pip install flask opencv-python numpy pillow
   ```

3. **Install optional AI detectors** (recommended)

   For HuggingFace AI Detector:
   ```bash
   pip install transformers torch torchvision
   ```

   For SVM Detector:
   ```bash
   pip install transformers torch torchvision scikit-learn joblib
   ```

### Running the Application

1. **Start the web server**
   ```bash
   python app.py
   ```

2. **Open your browser**

   Navigate to: `http://localhost:5000`

3. **Upload an image or video**
   - Drag & drop files, or
   - Click "Browse Files" to select, or
   - Paste an image URL

## 📁 Project Structure

```
DeepfakeDetect/
├── app.py                          # Flask web application
├── detector.py                     # CV-based detector (15 methods)
├── detector_hf.py                  # HuggingFace AI detector
├── detector_svm.py                 # SVM + ViT detector
├── static/
│   ├── style.css                  # Modern UI styles
│   └── script.js                  # Frontend JavaScript
├── templates/
│   └── index.html                 # Web interface
├── uploads/                        # Temporary upload folder
└── README.md                       # This file
```

## 🔬 How It Works

### CV Detector Methods

The CV detector uses 15 complementary analysis techniques:

1. **Frequency Analysis** - Detects unnatural frequency patterns in AI-generated images
2. **Channel Correlation** - Analyzes noise correlation across RGB channels
3. **Micro-Texture** - Examines fine skin details and pores
4. **Color Statistics** - Checks color distribution anomalies
5. **Noise Patterns** - Identifies artificial noise signatures
6. **Edge Quality** - Analyzes edge sharpness and ringing artifacts
7. **Compression Artifacts** - Detects JPEG compression patterns
8. **Symmetry** - Checks for unnaturally perfect facial symmetry
9. **Eye Region** - Analyzes iris patterns and reflections
10. **Teeth Region** - Examines dental details and mouth interior
11. **Hair Boundary** - Checks hair-skin transition naturalness
12. **Skin Pores** - Detects presence of natural skin texture
13. **Lighting Consistency** - Analyzes shadow and light patterns
14. **Blur Consistency** - Checks depth-of-field patterns
15. **Background Consistency** - Compares face and background noise

### AI Detector (HuggingFace)

- Uses a fine-tuned Vision Transformer (ViT)
- Pre-trained on extensive deepfake datasets
- Provides probability scores for real vs. fake classification
- Best for general-purpose detection

### SVM Detector

- Extracts 768-dimensional embeddings using ViT
- Trains Support Vector Machine for classification
- Can be customized for specific deepfake types
- Supports GridSearchCV for hyperparameter tuning

## 🎓 Training Custom SVM Models

You can train the SVM detector on your own datasets:

```python
from detector_svm import SVMDeepfakeDetector

# Initialize detector
detector = SVMDeepfakeDetector()

# Build dataset from folders
X, y = detector.build_dataset_from_folders(
    real_folder='path/to/real/images',
    fake_folder='path/to/fake/images'
)

# Train the SVM
detector.train(X, y, use_grid_search=True)

# Save the trained model
detector.save_model('svm_model.joblib')
```

The web app will automatically load `svm_model.joblib` if present.

## 📊 API Endpoints

### `GET /`
Returns the main web interface

### `POST /analyze`
Analyzes uploaded image/video file

**Parameters:**
- `file`: Image or video file
- `detector`: Detection method (`cv`, `hf`, or `svm`)

**Response:**
```json
{
  "verdict": "REAL" or "FAKE",
  "confidence": 85.5,
  "faces_detected": 1,
  "overall_score": 0.145,
  "face_analyses": [...],
  "detector_used": "cv"
}
```

### `POST /analyze-url`
Analyzes image from URL

**Body:**
```json
{
  "url": "https://example.com/image.jpg",
  "detector": "cv"
}
```

### `GET /api/status`
Returns detector availability status

## 🎨 Supported Formats

### Images
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)

### Videos
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

## ⚙️ Configuration

### Detector Selection

Choose the detector based on your needs:

| Detector | Speed | Accuracy | GPU Required | Trainable |
|----------|-------|----------|--------------|-----------|
| CV       | ⚡⚡⚡ Fast | ⭐⭐ Good | ❌ No | ❌ No |
| AI (HF)  | ⚡ Moderate | ⭐⭐⭐ Excellent | ✅ Optional | ❌ No |
| SVM      | ⚡⚡ Fast | ⭐⭐⭐ Excellent* | ✅ Optional | ✅ Yes |

*When trained on relevant datasets

### Video Analysis

Adjust frame sampling rate in `app.py`:
```python
results = detector.analyze_video(filepath, sample_rate=15)  # Analyze every 15th frame
```

Lower values = more frames analyzed = slower but more accurate

## 🛠️ Development

### Running in Development Mode

The app runs in debug mode by default:
```bash
python app.py
```

### Face Detection Model

The app uses YuNet face detector. If not present, it will:
1. Automatically download on first run (12MB)
2. Fall back to Haar Cascade if download fails

## ⚠️ Important Notes

### Educational Purpose
This is an educational tool. For critical decisions (legal, forensic, etc.), use professional-grade solutions.

### Limitations
- No detector is 100% accurate
- Sophisticated deepfakes may evade detection
- Results should be verified with multiple methods
- Quality and resolution affect accuracy

### Privacy
- All processing is done locally
- Files are temporarily stored and immediately deleted after analysis
- No data is sent to external servers (except for URL-based analysis)

## 📝 Requirements

### Core Dependencies
```
flask>=3.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

### Optional Dependencies

For AI Detector:
```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
```

For SVM Detector (includes AI detector requirements):
```
transformers>=4.30.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **HuggingFace** - Pre-trained AI models
- **YuNet** - Face detection model
- **Flask** - Web framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/DeepfakeDetect/issues) page
2. Create a new issue with detailed information
3. Provide sample images/videos if possible (ensure they don't contain private information)

## 🔮 Future Enhancements

- [ ] Batch processing for multiple files
- [ ] Export analysis reports (PDF/JSON)
- [ ] Real-time webcam analysis
- [ ] Additional AI models integration
- [ ] Confidence threshold customization
- [ ] Advanced visualization of detection features
- [ ] REST API documentation
- [ ] Docker containerization

---

Made with ❤️ using Python & Flask

**⚠️ Disclaimer:** This tool is for educational and research purposes only. Always verify results with professional forensic analysis for critical use cases.
