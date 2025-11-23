/**
 * Deepfake Detector - Frontend JavaScript
 * Handles file uploads, API calls, and result display
 */

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const urlBtn = document.getElementById('urlBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const uploadSection = document.querySelector('.upload-section');

// Image data storage
let currentImages = {
    original: null,
    annotated: null
};

// Get selected detector type
function getSelectedDetector() {
    const selected = document.querySelector('input[name="detector"]:checked');
    return selected ? selected.value : 'cv';
}

// ==========================================
// FILE UPLOAD HANDLING
// ==========================================

// Click to upload
dropZone.addEventListener('click', () => fileInput.click());

// File selected via input
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// URL analysis
urlBtn.addEventListener('click', () => {
    const url = urlInput.value.trim();
    if (url) {
        analyzeUrl(url);
    }
});

urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const url = urlInput.value.trim();
        if (url) {
            analyzeUrl(url);
        }
    }
});

// New analysis button
newAnalysisBtn.addEventListener('click', resetToUpload);

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tab = btn.dataset.tab;
        const previewImage = document.getElementById('previewImage');

        if (tab === 'original' && currentImages.original) {
            previewImage.src = `data:image/jpeg;base64,${currentImages.original}`;
        } else if (currentImages.annotated) {
            previewImage.src = `data:image/jpeg;base64,${currentImages.annotated}`;
        }
    });
});

// ==========================================
// FILE HANDLING
// ==========================================

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp',
                       'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];

    if (!validTypes.some(type => file.type.startsWith(type.split('/')[0]))) {
        alert('Please upload a valid image or video file.');
        return;
    }

    // Check file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
        alert('File too large. Maximum size is 50MB.');
        return;
    }

    analyzeFile(file);
}

// ==========================================
// API CALLS
// ==========================================

async function analyzeFile(file) {
    showLoading();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('detector', getSelectedDetector());

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const results = await response.json();

        if (results.error) {
            throw new Error(results.error);
        }

        displayResults(results);
    } catch (error) {
        hideLoading();
        alert(`Analysis failed: ${error.message}`);
    }
}

async function analyzeUrl(url) {
    showLoading();

    try {
        const response = await fetch('/analyze-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url, detector: getSelectedDetector() })
        });

        const results = await response.json();

        if (results.error) {
            throw new Error(results.error);
        }

        displayResults(results);
    } catch (error) {
        hideLoading();
        alert(`Analysis failed: ${error.message}`);
    }
}

// ==========================================
// RESULTS DISPLAY
// ==========================================

function displayResults(results) {
    hideLoading();

    // Store images
    currentImages.original = results.original_image;
    currentImages.annotated = results.annotated_image;

    // Show results section
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'grid';

    // Display preview image
    const previewImage = document.getElementById('previewImage');
    if (results.annotated_image) {
        previewImage.src = `data:image/jpeg;base64,${results.annotated_image}`;
    }

    // Update detector badge
    const detectorBadge = document.getElementById('detectorBadge');
    const detectorUsed = results.detector_used || 'cv';
    if (detectorUsed === 'hf') {
        detectorBadge.textContent = 'AI Detector';
        detectorBadge.classList.add('hf');
    } else {
        detectorBadge.textContent = 'CV Detector';
        detectorBadge.classList.remove('hf');
    }

    // Update verdict
    const verdictCard = document.getElementById('verdictCard');
    const verdictText = document.getElementById('verdictText');
    const verdict = results.verdict || 'Unknown';

    verdictText.textContent = verdict;

    // Set verdict styling
    verdictCard.classList.remove('real', 'fake', 'uncertain');
    if (verdict.includes('REAL')) {
        verdictCard.classList.add('real');
    } else if (verdict.includes('FAKE')) {
        verdictCard.classList.add('fake');
    } else {
        verdictCard.classList.add('uncertain');
    }

    // Update confidence
    const confidence = results.confidence || 0;
    document.getElementById('confidenceValue').textContent = confidence.toFixed(1);

    // Animate confidence bar
    setTimeout(() => {
        document.getElementById('confidenceBar').style.width = `${confidence}%`;
    }, 100);

    // Update faces count
    document.getElementById('facesCount').textContent = results.faces_detected || 0;

    // Update detailed scores
    if (results.face_analyses && results.face_analyses.length > 0) {
        const face = results.face_analyses[0];

        // HF detector provides different scores, use fallbacks
        updateScoreBar('freq', face.frequency_score || face.deepfake_probability || 0);
        updateScoreBar('consistency', face.consistency_score || face.texture_score || 0);
        updateScoreBar('quality', face.quality_score || face.ai_indicators || 0);
    } else {
        // No faces detected - reset scores
        updateScoreBar('freq', 0);
        updateScoreBar('consistency', 0);
        updateScoreBar('quality', 0);
    }

    // Update explanation
    const explanation = generateExplanation(results);
    document.getElementById('explanationText').textContent = explanation;
}

function updateScoreBar(id, value) {
    const bar = document.getElementById(`${id}Bar`);
    const valueEl = document.getElementById(`${id}Value`);

    const percentage = (value || 0) * 100;

    // Update value display
    valueEl.textContent = value ? value.toFixed(2) : '-';

    // Animate bar
    setTimeout(() => {
        bar.style.width = `${percentage}%`;

        // Set color based on value
        bar.classList.remove('low', 'medium', 'high');
        if (value < 0.3) {
            bar.classList.add('low');
        } else if (value < 0.6) {
            bar.classList.add('medium');
        } else {
            bar.classList.add('high');
        }
    }, 100);
}

function generateExplanation(results) {
    const verdict = results.verdict || 'Unknown';
    const confidence = results.confidence || 0;
    const faces = results.faces_detected || 0;
    const isHF = results.detector_used === 'hf';

    if (faces === 0 && !results.full_image_analysis) {
        return 'No faces were detected in this image. The detector needs visible faces to analyze for deepfakes. Try uploading an image with clear, visible faces.';
    }

    let explanation = '';
    const detectorName = isHF ? 'AI model (Hugging Face)' : 'computer vision analysis';

    if (verdict.includes('REAL')) {
        explanation = `This image appears to be authentic with ${confidence.toFixed(1)}% confidence (using ${detectorName}). `;
        if (isHF) {
            explanation += 'The deep learning model found patterns consistent with authentic images. ';
        } else {
            explanation += 'The analysis found natural frequency patterns, consistent face boundaries, and no obvious manipulation artifacts. ';
        }
        explanation += 'However, sophisticated deepfakes may still evade detection.';
    } else if (verdict.includes('FAKE')) {
        explanation = `This image shows signs of manipulation with ${confidence.toFixed(1)}% confidence (using ${detectorName}). `;

        if (results.face_analyses && results.face_analyses.length > 0) {
            const face = results.face_analyses[0];
            const issues = [];

            if (isHF) {
                if (face.deepfake_probability > 0.6) {
                    issues.push('AI-generated facial features');
                }
            } else {
                if (face.frequency_score > 0.4) {
                    issues.push('unusual frequency patterns typical of AI generation');
                }
                if (face.consistency_score > 0.4) {
                    issues.push('inconsistencies at face boundaries');
                }
                if (face.quality_score > 0.4) {
                    issues.push('quality anomalies suggesting manipulation');
                }
            }

            if (issues.length > 0) {
                explanation += `Detected: ${issues.join(', ')}. `;
            }
        }

        // Check for mixed faces
        if (results.has_mixed_faces && results.comparison_note) {
            explanation += `Note: ${results.comparison_note}. `;
        }

        explanation += 'Consider verifying with additional forensic tools.';
    } else {
        explanation = `The results are inconclusive (${confidence.toFixed(1)}% confidence). `;
        explanation += 'The image shows some characteristics that could indicate manipulation, but not enough to make a definitive determination. ';
        if (!isHF) {
            explanation += 'Try using the AI Detector for potentially better results. ';
        }
        explanation += 'Use professional forensic tools for critical decisions.';
    }

    return explanation;
}

// ==========================================
// UI HELPERS
// ==========================================

function showLoading() {
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

function resetToUpload() {
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';

    // Reset file input
    fileInput.value = '';
    urlInput.value = '';

    // Reset images
    currentImages = { original: null, annotated: null };

    // Reset scores
    document.getElementById('confidenceBar').style.width = '0%';
    document.getElementById('freqBar').style.width = '0%';
    document.getElementById('consistencyBar').style.width = '0%';
    document.getElementById('qualityBar').style.width = '0%';

    // Reset tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === 'annotated') {
            btn.classList.add('active');
        }
    });
}

// ==========================================
// INITIALIZATION
// ==========================================

console.log('Deepfake Detector loaded successfully!');
