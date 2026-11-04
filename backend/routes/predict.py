# =============================================================================
# AgriLite-Hybrid Backend
# routes/predict.py - Disease Prediction Routes
# 
# Handles plant disease prediction endpoints including image upload,
# processing, inference, and result generation with Grad-CAM heatmaps.
# =============================================================================

import os
import json
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request

from extensions import db, limiter
from models import DetectionHistory
from utils import (
    validate_crop, 
    allowed_file, 
    generate_unique_filename,
    success_response, 
    error_response
)
from decorators import handle_db_errors

# Create blueprint
predict_bp = Blueprint('predict', __name__)


# =============================================================================
# Main Prediction Endpoint
# =============================================================================

@predict_bp.route('/', methods=['POST'])
@limiter.limit("30 per minute")
@handle_db_errors
def predict():
    """
    Analyze plant image for disease detection.
    
    This is the main prediction endpoint that:
    1. Receives crop type and image
    2. Preprocesses the image
    3. Runs inference through the ML model
    4. Generates Grad-CAM heatmap
    5. Returns disease, severity, confidence, and treatment
    
    Request:
        Content-Type: multipart/form-data
        
        Fields:
            crop (str): Crop type (tomato, chilli) - required
            image (file): Image file (jpg, jpeg, png, webp) - required
            temperature (float): Environmental temperature in °C - optional
            humidity (float): Environmental humidity in % - optional
            save_history (bool): Whether to save to history (requires auth) - optional
            
    Returns:
        200: Prediction results with disease, severity, treatment, and heatmap
        400: Validation error
        500: Prediction error
        
    Response Example:
        {
            "success": true,
            "data": {
                "crop": "tomato",
                "disease": "Early Blight",
                "severity": "medium",
                "confidence": 98.7,
                "treatment": ["Remove infected leaves...", ...],
                "heatmap_base64": "data:image/jpeg;base64,...",
                "original_image_base64": "data:image/jpeg;base64,...",
                "environmental_risk": "normal"
            }
        }
    """
    # ==========================================================================
    # Validate Request
    # ==========================================================================
    
    # Check for crop type
    crop = request.form.get('crop', '').lower().strip()
    if not crop:
        return error_response(
            'Crop type is required',
            details={'field': 'crop', 'allowed': ['tomato', 'chilli']},
            status_code=400
        )
    
    if not validate_crop(crop):
        return error_response(
            f"Invalid crop type: '{crop}'",
            details={'allowed': ['tomato', 'chilli']},
            status_code=400
        )
    
    # Check for image file
    if 'image' not in request.files:
        return error_response(
            'Image file is required',
            details={'field': 'image'},
            status_code=400
        )
    
    file = request.files['image']
    
    if file.filename == '':
        return error_response('No file selected', status_code=400)
    
    if not allowed_file(file.filename):
        return error_response(
            'Invalid file type',
            details={'allowed': ['png', 'jpg', 'jpeg', 'webp']},
            status_code=400
        )
    
    # Get optional environmental data
    temperature = request.form.get('temperature', type=float)
    humidity = request.form.get('humidity', type=float)
    save_history = request.form.get('save_history', 'false').lower() == 'true'
    
    # ==========================================================================
    # Process Image and Run Prediction
    # ==========================================================================
    
    try:
        # Get ML service from app config
        ml_service = current_app.config.get('ML_SERVICE')
        
        if ml_service is None:
            # Import and create service if not initialized
            from services.ml_service import MLService
            ml_service = MLService()
        
        # Read image data
        image_data = file.read()
        file.seek(0)  # Reset file pointer
        
        # Run prediction
        result = ml_service.predict(
            image_data=image_data,
            crop=crop,
            temperature=temperature,
            humidity=humidity
        )
        
        if not result.get('success', False):
            return error_response(
                result.get('error', 'Prediction failed'),
                status_code=400
            )
        
        # =======================================================================
        # Save to History (if authenticated and requested)
        # =======================================================================
        
        user_id = None
        try:
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
            if user_id:
                user_id = int(user_id)
        except:
            pass
        
        if save_history and user_id:
            history_entry = DetectionHistory(
                user_id=user_id,
                crop=crop,
                disease=result['disease'],
                severity=result['severity'],
                confidence=result['confidence'],
                treatment=json.dumps(result['treatment']),
                original_image=result.get('original_image_base64', ''),
                heatmap_image=result.get('heatmap_base64', ''),
                temperature=temperature,
                humidity=humidity,
                environmental_risk=result.get('environmental_risk'),
                device_type='web'
            )
            db.session.add(history_entry)
            db.session.commit()
            result['history_id'] = history_entry.id
        
        current_app.logger.info(
            f"Prediction completed: {crop} - {result['disease']} "
            f"({result['confidence']*100:.1f}%)"
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return error_response(
            'An error occurred during prediction',
            details=str(e) if current_app.debug else None,
            status_code=500
        )


# =============================================================================
# Quick Prediction (Base64 Image Input)
# =============================================================================

@predict_bp.route('/base64', methods=['POST'])
@limiter.limit("30 per minute")
@handle_db_errors
def predict_base64():
    """
    Analyze plant image from base64 encoded string.
    
    Alternative endpoint for frontend applications that capture
    images via camera (getUserMedia) and send as base64.
    
    Request Body:
        {
            "crop": "tomato",
            "image_base64": "data:image/jpeg;base64,...",
            "temperature": 28.5,  // optional
            "humidity": 65.0      // optional
        }
        
    Returns:
        Same as /predict endpoint
    """
    data = request.get_json()
    
    if not data:
        return error_response('Request body is required', status_code=400)
    
    # Validate crop
    crop = data.get('crop', '').lower().strip()
    if not crop or not validate_crop(crop):
        return error_response(
            'Valid crop type is required',
            details={'allowed': ['tomato', 'chilli']},
            status_code=400
        )
    
    # Validate image
    image_base64 = data.get('image_base64', '')
    if not image_base64:
        return error_response('Image base64 data is required', status_code=400)
    
    # Get optional data
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    
    try:
        import base64
        
        # Remove data URI prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to bytes
        image_data = base64.b64decode(image_base64)
        
        # Get ML service
        ml_service = current_app.config.get('ML_SERVICE')
        if ml_service is None:
            from services.ml_service import MLService
            ml_service = MLService()
        
        # Run prediction
        result = ml_service.predict(
            image_data=image_data,
            crop=crop,
            temperature=temperature,
            humidity=humidity
        )
        
        if not result.get('success', False):
            return error_response(
                result.get('error', 'Prediction failed'),
                status_code=400
            )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Base64 prediction error: {e}")
        return error_response(
            'An error occurred during prediction',
            status_code=500
        )


# =============================================================================
# Get Treatment Information
# =============================================================================

@predict_bp.route('/treatment', methods=['GET'])
@limiter.limit("100 per minute")
def get_treatment():
    """
    Get treatment recommendations for a specific disease and severity.
    
    Query Parameters:
        disease (str): Disease name (required)
        severity (str): Severity level - mild, medium, severe (optional, default: medium)
        
    Returns:
        200: Treatment recommendations
        400: Missing disease parameter
        404: Disease not found
    """
    disease = request.args.get('disease', '').strip()
    severity = request.args.get('severity', 'medium').lower().strip()
    
    if not disease:
        return error_response('Disease name is required', status_code=400)
    
    if severity not in ['mild', 'medium', 'severe']:
        severity = 'medium'
    
    # Get ML service for treatment data
    ml_service = current_app.config.get('ML_SERVICE')
    if ml_service is None:
        from services.ml_service import MLService
        ml_service = MLService()
    
    treatment = ml_service.get_treatment(disease, severity)
    
    if not treatment:
        return error_response(
            f"Treatment not found for disease: '{disease}'",
            status_code=404
        )
    
    return jsonify({
        'success': True,
        'data': {
            'disease': disease,
            'severity': severity,
            'treatment': treatment
        }
    }), 200
