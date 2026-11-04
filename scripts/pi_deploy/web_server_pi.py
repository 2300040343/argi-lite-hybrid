#!/usr/bin/env python3
"""
=============================================================================
AgriLite-Hybrid - Raspberry Pi Web Server
web_server_pi.py - Lightweight Flask server for local network access

Provides a simple web interface and API for plant disease detection
on Raspberry Pi over local network.
=============================================================================
"""

import argparse
import base64
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from PIL import Image

# Import predictor from local module
from predict_pi import AgriLitePredictor, capture_image_from_camera

# =============================================================================
# Flask Application
# =============================================================================

app = Flask(__name__)
CORS(app)

# Initialize predictor (will be set in main)
predictor = None

# Supported crops
SUPPORTED_CROPS = ["tomato", "chilli"]

# =============================================================================
# HTML Templates
# =============================================================================

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriLite-Hybrid - Raspberry Pi</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #f0f9f4; min-height: 100vh; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 30px 0; }
        header h1 { color: #166534; font-size: 1.8rem; margin-bottom: 8px; }
        header p { color: #6b7280; }
        .card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .form-group { margin-bottom: 16px; }
        label { display: block; font-weight: 500; color: #374151; margin-bottom: 8px; }
        select, input[type="file"] { width: 100%; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 16px; }
        select:focus { outline: none; border-color: #16a34a; }
        .btn { width: 100%; padding: 14px; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-primary { background: #16a34a; color: white; }
        .btn-primary:hover { background: #15803d; }
        .btn-secondary { background: #e5e7eb; color: #374151; margin-top: 10px; }
        .btn-secondary:hover { background: #d1d5db; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        #preview { display: none; max-width: 100%; border-radius: 8px; margin: 16px 0; }
        #result { display: none; }
        .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .severity { padding: 4px 12px; border-radius: 999px; font-size: 14px; font-weight: 500; }
        .severity-healthy { background: #dcfce7; color: #166534; }
        .severity-mild { background: #fef9c3; color: #854d0e; }
        .severity-moderate { background: #fed7aa; color: #9a3412; }
        .severity-severe { background: #fecaca; color: #991b1b; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #6b7280; }
        .stat-value { font-weight: 600; color: #111827; }
        .treatment { background: #f0fdf4; border-left: 3px solid #16a34a; padding: 12px; margin-top: 8px; border-radius: 0 8px 8px 0; }
        .treatment-title { font-weight: 600; color: #166534; margin-bottom: 4px; }
        .treatment-detail { font-size: 14px; color: #4b5563; }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { width: 40px; height: 40px; border: 3px solid #e5e7eb; border-top-color: #16a34a; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error { background: #fef2f2; color: #991b1b; padding: 12px; border-radius: 8px; margin-bottom: 16px; display: none; }
        footer { text-align: center; padding: 20px; color: #6b7280; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌿 AgriLite-Hybrid</h1>
            <p>Plant Disease Detection on Raspberry Pi</p>
        </header>

        <div class="card">
            <div class="error" id="error"></div>
            
            <form id="uploadForm">
                <div class="form-group">
                    <label for="crop">Select Crop</label>
                    <select id="crop" name="crop" required>
                        <option value="">Choose crop...</option>
                        <option value="tomato">🍅 Tomato</option>
                        <option value="chilli">🌶️ Chilli</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="image">Upload Leaf Image</label>
                    <input type="file" id="image" name="image" accept="image/*" capture="environment" required>
                </div>

                <img id="preview" alt="Preview">

                <button type="submit" class="btn btn-primary" id="submitBtn">Analyze Image</button>
                <button type="button" class="btn btn-secondary" id="cameraBtn" onclick="captureFromCamera()">
                    📷 Capture from Camera
                </button>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing image...</p>
        </div>

        <div class="card" id="result">
            <div class="result-header">
                <h2 id="diseaseName">Disease Name</h2>
                <span class="severity" id="severityBadge">Severity</span>
            </div>

            <div class="stat">
                <span class="stat-label">Crop</span>
                <span class="stat-value" id="cropName">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Confidence</span>
                <span class="stat-value" id="confidence">-</span>
            </div>
            <div class="stat">
                <span class="stat-label">Inference Time</span>
                <span class="stat-value" id="inferenceTime">-</span>
            </div>

            <h3 style="margin-top: 20px; margin-bottom: 12px; color: #374151;">Treatment Recommendations</h3>
            <div id="treatments"></div>

            <button onclick="resetForm()" class="btn btn-secondary" style="margin-top: 20px;">
                Analyze Another Image
            </button>
        </div>

        <footer>
            <p>Running on Raspberry Pi • AgriLite-Hybrid v1.0</p>
        </footer>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const imageInput = document.getElementById('image');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const errorDiv = document.getElementById('error');

        // Show image preview
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        // Form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            
            errorDiv.style.display = 'none';
            form.style.display = 'none';
            loading.style.display = 'block';
            result.style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    displayResult(data);
                } else {
                    throw new Error(data.error || 'Prediction failed');
                }
            } catch (err) {
                errorDiv.textContent = err.message;
                errorDiv.style.display = 'block';
                form.style.display = 'block';
            } finally {
                loading.style.display = 'none';
            }
        });

        // Capture from camera
        async function captureFromCamera() {
            errorDiv.style.display = 'none';
            
            const crop = document.getElementById('crop').value;
            if (!crop) {
                errorDiv.textContent = 'Please select a crop first';
                errorDiv.style.display = 'block';
                return;
            }

            form.style.display = 'none';
            loading.style.display = 'block';

            try {
                const response = await fetch('/api/capture?crop=' + crop, {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    displayResult(data);
                } else {
                    throw new Error(data.error || 'Capture failed');
                }
            } catch (err) {
                errorDiv.textContent = err.message;
                errorDiv.style.display = 'block';
                form.style.display = 'block';
            } finally {
                loading.style.display = 'none';
            }
        }

        // Display result
        function displayResult(data) {
            document.getElementById('diseaseName').textContent = data.disease;
            document.getElementById('cropName').textContent = data.crop.charAt(0).toUpperCase() + data.crop.slice(1);
            document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('inferenceTime').textContent = data.inference_time_ms + ' ms';

            const badge = document.getElementById('severityBadge');
            badge.textContent = data.severity.charAt(0).toUpperCase() + data.severity.slice(1);
            badge.className = 'severity severity-' + data.severity;

            // Treatments
            const treatmentsDiv = document.getElementById('treatments');
            treatmentsDiv.innerHTML = '';
            if (data.treatment && data.treatment.length > 0) {
                data.treatment.forEach(t => {
                    treatmentsDiv.innerHTML += `
                        <div class="treatment">
                            <div class="treatment-title">[${t.type}] ${t.name}</div>
                            ${t.dosage && t.dosage !== 'N/A' ? '<div class="treatment-detail">Dosage: ' + t.dosage + '</div>' : ''}
                        </div>
                    `;
                });
            } else {
                treatmentsDiv.innerHTML = '<p style="color: #6b7280;">No specific treatment needed for healthy plants.</p>';
            }

            result.style.display = 'block';
        }

        // Reset form
        function resetForm() {
            form.reset();
            preview.style.display = 'none';
            result.style.display = 'none';
            form.style.display = 'block';
        }
    </script>
</body>
</html>
"""

# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Serve main web interface."""
    return render_template_string(INDEX_HTML)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/crops')
def get_crops():
    """Get list of supported crops."""
    return jsonify({
        "success": True,
        "data": SUPPORTED_CROPS
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict disease from uploaded image.
    
    Expected form data:
    - crop: string (tomato, chilli)
    - image: file (JPEG, PNG)
    """
    global predictor
    
    try:
        # Validate crop
        crop = request.form.get('crop', '').lower()
        if crop not in SUPPORTED_CROPS:
            return jsonify({
                "success": False,
                "error": f"Invalid crop. Must be one of: {', '.join(SUPPORTED_CROPS)}"
            }), 400
        
        # Validate image
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image selected"
            }), 400
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image_file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            # Run prediction
            result = predictor.predict(temp_path, crop)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict/base64', methods=['POST'])
def predict_base64():
    """
    Predict disease from base64 encoded image.
    
    Expected JSON:
    {
        "crop": "tomato",
        "image_base64": "base64_encoded_image_data"
    }
    """
    global predictor
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Validate crop
        crop = data.get('crop', '').lower()
        if crop not in SUPPORTED_CROPS:
            return jsonify({
                "success": False,
                "error": f"Invalid crop. Must be one of: {', '.join(SUPPORTED_CROPS)}"
            }), 400
        
        # Decode base64 image
        image_b64 = data.get('image_base64', '')
        if not image_b64:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        # Decode and save to temp file
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, 'JPEG')
            temp_path = tmp.name
        
        try:
            # Run prediction
            result = predictor.predict(temp_path, crop)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/capture', methods=['POST'])
def capture_and_predict():
    """
    Capture image from Pi camera and predict disease.
    
    Query parameters:
    - crop: string (tomato, chilli)
    """
    global predictor
    
    try:
        # Validate crop
        crop = request.args.get('crop', '').lower()
        if crop not in SUPPORTED_CROPS:
            return jsonify({
                "success": False,
                "error": f"Invalid crop. Must be one of: {', '.join(SUPPORTED_CROPS)}"
            }), 400
        
        # Capture image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = capture_image_from_camera(tmp.name)
        
        try:
            # Run prediction
            result = predictor.predict(temp_path, crop)
            return jsonify(result)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    global predictor
    
    parser = argparse.ArgumentParser(
        description="AgriLite-Hybrid Raspberry Pi Web Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to TFLite model file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("Initializing AgriLite-Hybrid predictor...")
    if args.model:
        predictor = AgriLitePredictor(model_path=Path(args.model))
    else:
        predictor = AgriLitePredictor()
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname + ".local")
    except:
        local_ip = socket.gethostbyname(hostname)
    
    print(f"\n{'='*60}")
    print(f"AgriLite-Hybrid Web Server")
    print(f"{'='*60}")
    print(f"Local URL:   http://localhost:{args.port}")
    print(f"Network URL: http://{local_ip}:{args.port}")
    print(f"{'='*60}\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()
