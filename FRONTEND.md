# Frontend Documentation

## Overview

The Deepfake Detector web interface is built with modern web technologies to provide an elegant, user-friendly experience for analyzing images and videos.

## Technologies Used

### 1. HTML5
- **File:** `templates/index.html`
- Semantic HTML structure
- Form handling for file uploads
- Responsive layout containers
- SVG graphics for icons and logo

### 2. CSS3
- **File:** `static/style.css`
- Modern CSS features:
  - CSS Custom Properties (variables) for theming
  - Flexbox and Grid layouts
  - CSS animations and transitions
  - Gradient backgrounds
  - Box shadows and glow effects
  - Media queries for responsiveness

### 3. JavaScript (Vanilla)
- **File:** `static/script.js`
- No frameworks - pure JavaScript for better performance
- Features:
  - Drag & drop file upload
  - AJAX requests with Fetch API
  - Dynamic DOM manipulation
  - Tab switching for image preview
  - Real-time result updates

### 4. Fonts
- **Poppins** - Modern sans-serif font for UI text
- **JetBrains Mono** - Monospace font for technical details
- Loaded from Google Fonts CDN

## Design Features

### Color Scheme
```css
Primary Color: #6366f1 (Indigo)
Success Color: #10b981 (Green)
Danger Color: #ef4444 (Red)
Background: #0a0f1a (Dark blue-black)
Cards: #111827 (Dark gray)
```

### Visual Effects
- **Gradient Backgrounds** - Subtle radial gradients for depth
- **Glow Effects** - Shadow glows on interactive elements
- **Smooth Animations** - 0.3s transitions on hover/click
- **Animated Logo** - Pulsing gradient stroke animation
- **Glassmorphism** - Semi-transparent cards with blur

### SVG Icons
All icons are custom SVG graphics instead of icon fonts:
- Lightning bolt (CV Detector)
- Robot (AI Detector)
- Diamond (SVM Detector)
- Upload arrows
- Search icon
- Link icon
- Chart bars
- And more...

## Key Components

### 1. Header
- Animated logo with gradient stroke
- App title with tagline
- Clean, centered layout

### 2. Detector Selector
- Three detector options (CV, AI, SVM)
- Radio button cards with hover effects
- Visual indicators for availability
- Automatic disable state for unavailable detectors

### 3. Upload Section
- Drag & drop zone
- File browser button
- Supported file types display
- URL input option
- Visual feedback on hover/drag

### 4. Results Section
- Image preview with tabs (Analyzed/Original)
- Verdict card with color-coded results
- Confidence meter with animated bar
- Analysis breakdown scores
- Face detection count
- Explanation card
- "Analyze Another" button

### 5. Loading Overlay
- Full-screen overlay during analysis
- Animated spinner icon
- Loading text with subtext
- Semi-transparent backdrop

## Interactive Features

### Drag & Drop Upload
```javascript
- Drag over: Visual highlight
- Drop: Auto-upload and analyze
- Supports images and videos
```

### Tab Switching
```javascript
- Original image view
- Analyzed image with face boxes
- Smooth transitions
```

### Result Display
```javascript
- Color-coded verdicts (green=real, red=fake)
- Animated confidence bars
- Per-face analysis details
- Dynamic explanations
```

## Responsive Design

The interface adapts to different screen sizes:
- Desktop: Full layout with side-by-side panels
- Tablet: Stacked layout with touch-friendly buttons
- Mobile: Single column, optimized spacing

## File Upload Support

### Supported Formats
**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)

**Videos:**
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

### Upload Methods
1. Drag and drop files
2. Click "Browse Files" button
3. Paste image URL

## API Integration

### Endpoints Used
- `GET /` - Load main page
- `POST /analyze` - Upload and analyze file
- `POST /analyze-url` - Analyze from URL
- `GET /api/status` - Check detector availability

### Response Handling
```javascript
{
  "verdict": "REAL" or "FAKE",
  "confidence": 85.5,
  "faces_detected": 1,
  "annotated_image": "base64_string",
  "face_analyses": [...],
  "detector_used": "cv"
}
```

## User Experience

### Visual Feedback
- Hover effects on all interactive elements
- Loading states during analysis
- Error messages for invalid files
- Success/failure color coding

### Accessibility
- Semantic HTML tags
- Clear button labels
- High contrast text
- Keyboard navigation support

### Performance
- Lazy loading of detectors
- Client-side file validation
- Optimized animations (60fps)
- Compressed image encoding

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Future Enhancements

- Dark/Light theme toggle
- Batch upload multiple files
- Download analysis reports
- Real-time webcam analysis
- Comparison mode (side-by-side)
- Zoom controls for image preview

---

**Built with modern web standards for a smooth, professional user experience.**
