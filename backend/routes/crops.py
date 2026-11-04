# =============================================================================
# AgriLite-Hybrid Backend
# routes/crops.py - Crops Routes
# 
# Handles crop information endpoints including listing available crops
# and their associated diseases and metadata.
# =============================================================================

import os
import json
from flask import Blueprint, jsonify, current_app

from extensions import limiter
from utils import success_response, error_response

# Create blueprint
crops_bp = Blueprint('crops', __name__)


# =============================================================================
# Crop Data Configuration
# =============================================================================

# Default crop data (used if class_labels.json is not available)
DEFAULT_CROPS = {
    'tomato': {
        'name': 'tomato',
        'display_name': 'Tomato',
        'description': 'Widely cultivated crop affected by various bacterial and fungal diseases. Proper diagnosis improves treatment success.',
        'image': '/images/crops/tomato.jpg',
        'diseases': [
            'Bacterial Spot',
            'Early Blight',
            'Late Blight',
            'Leaf Mold',
            'Septoria Leaf Spot',
            'Spider Mites (Two-spotted)',
            'Target Spot',
            'Yellow Leaf Curl Virus',
            'Mosaic Virus',
            'Healthy',
            'Powdery Mildew'
        ],
        'color': '#EF4444'  # Red
    },
    'chilli': {
        'name': 'chilli',
        'display_name': 'Chilli',
        'description': 'Spice crop prone to viral infections and fungal diseases. Quick identification prevents spread.',
        'image': '/images/crops/chilli.jpg',
        'diseases': [
            'Anthracnose',
            'Damping Off',
            'Leaf Curl Virus',
            'Leaf Spot',
            'Veinal Mottle Virus',
            'Whitefly',
            'Yellowish',
            'Healthy'
        ],
        'color': '#F97316'  # Orange
    }
}


def load_crop_data():
    """
    Load crop data from class_labels.json or use defaults.
    
    Returns:
        dict: Crop data dictionary
    """
    try:
        model_path = current_app.config.get('MODEL_PATH', '../models')
        labels_file = os.path.join(model_path, 'class_labels.json')
        
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                data = json.load(f)
                
            # Transform to expected format
            crops = {}
            for crop_name, crop_info in data.get('crops', {}).items():
                crops[crop_name] = {
                    'name': crop_name,
                    'display_name': crop_info.get('name', crop_name.title()),
                    'description': crop_info.get('description', ''),
                    'image': crop_info.get('image', f'/images/crops/{crop_name}.jpg'),
                    'diseases': crop_info.get('diseases', []),
                    'color': DEFAULT_CROPS.get(crop_name, {}).get('color', '#6B7280')
                }
            return crops
            
    except Exception as e:
        current_app.logger.warning(f"Could not load class_labels.json: {e}")
    
    return DEFAULT_CROPS


# =============================================================================
# Get All Crops
# =============================================================================

@crops_bp.route('/', methods=['GET'])
@limiter.limit("100 per minute")
def get_all_crops():
    """
    Get list of all supported crops with their metadata.
    
    Returns:
        200: List of crops with names, descriptions, and diseases
        
    Response Example:
        {
            "success": true,
            "data": {
                "crops": [
                    {
                        "name": "tomato",
                        "display_name": "Tomato",
                        "description": "...",
                        "image": "/images/crops/tomato.jpg",
                        "diseases": ["Early Blight", ...],
                        "color": "#EF4444"
                    },
                    ...
                ],
                "total": 2
            }
        }
    """
    crops = load_crop_data()
    
    crops_list = list(crops.values())
    
    return jsonify({
        'success': True,
        'data': {
            'crops': crops_list,
            'total': len(crops_list)
        }
    }), 200


# =============================================================================
# Get Single Crop Details
# =============================================================================

@crops_bp.route('/<string:crop_name>', methods=['GET'])
@limiter.limit("100 per minute")
def get_crop(crop_name):
    """
    Get detailed information about a specific crop.
    
    Args:
        crop_name: Name of the crop (tomato, chilli)
        
    Returns:
        200: Crop details including diseases and treatments
        404: Crop not found
        
    Response Example:
        {
            "success": true,
            "data": {
                "name": "tomato",
                "display_name": "Tomato",
                "description": "...",
                "image": "/images/crops/tomato.jpg",
                "diseases": [...],
                "disease_count": 4
            }
        }
    """
    crop_name = crop_name.lower().strip()
    crops = load_crop_data()
    
    if crop_name not in crops:
        return error_response(
            f"Crop '{crop_name}' not found",
            details={
                'available_crops': list(crops.keys())
            },
            status_code=404
        )
    
    crop = crops[crop_name]
    crop['disease_count'] = len(crop.get('diseases', []))
    
    return jsonify({
        'success': True,
        'data': crop
    }), 200


# =============================================================================
# Get Crop Diseases
# =============================================================================

@crops_bp.route('/<string:crop_name>/diseases', methods=['GET'])
@limiter.limit("100 per minute")
def get_crop_diseases(crop_name):
    """
    Get list of diseases for a specific crop.
    
    Args:
        crop_name: Name of the crop
        
    Returns:
        200: List of diseases for the crop
        404: Crop not found
    """
    crop_name = crop_name.lower().strip()
    crops = load_crop_data()
    
    if crop_name not in crops:
        return error_response(
            f"Crop '{crop_name}' not found",
            status_code=404
        )
    
    diseases = crops[crop_name].get('diseases', [])
    
    # Add disease details if available
    disease_details = []
    for disease in diseases:
        disease_info = {
            'name': disease,
            'is_healthy': disease.lower() == 'healthy'
        }
        disease_details.append(disease_info)
    
    return jsonify({
        'success': True,
        'data': {
            'crop': crop_name,
            'diseases': disease_details,
            'total': len(disease_details)
        }
    }), 200


# =============================================================================
# Get Supported Crops List
# =============================================================================

@crops_bp.route('/names', methods=['GET'])
@limiter.limit("200 per minute")
def get_crop_names():
    """
    Get simple list of supported crop names.
    
    Returns:
        200: Array of crop names
        
    Response Example:
        {
            "success": true,
            "data": ["tomato", "chilli"]
        }
    """
    crops = load_crop_data()
    
    return jsonify({
        'success': True,
        'data': list(crops.keys())
    }), 200
