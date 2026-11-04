# =============================================================================
# AgriLite-Hybrid Backend
# utils.py - Utility Functions
# 
# Common utility functions used across the application including
# validation, file handling, and helper functions.
# =============================================================================

import re
import os
import base64
import hashlib
import random
import string
from datetime import datetime
from functools import wraps
from flask import request, jsonify, current_app
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
from werkzeug.utils import secure_filename

from extensions import db
from models import User


# =============================================================================
# Validation Functions
# =============================================================================

def validate_email(email: str) -> bool:
    """
    Validate email format using regex pattern.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength requirements.
    
    Requirements:
    - At least 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 number
    
    Args:
        password: Password to validate
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"


def validate_crop(crop: str) -> bool:
    """
    Validate if crop is in the list of supported crops.
    
    Args:
        crop: Crop name to validate
        
    Returns:
        bool: True if crop is supported, False otherwise
    """
    supported_crops = ['tomato', 'chilli']
    return crop.lower() in supported_crops


# =============================================================================
# File Handling Functions
# =============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has an allowed extension.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        bool: True if extension is allowed, False otherwise
    """
    allowed_extensions = current_app.config.get(
        'ALLOWED_EXTENSIONS',
        {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    )
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing special characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove special characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    return secure_filename(filename)


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename with timestamp and random suffix.
    
    Args:
        original_filename: Original filename with extension
        
    Returns:
        str: Unique filename
    """
    # Get file extension
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
    
    # Generate unique name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    return f"{timestamp}_{random_suffix}.{ext}"


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 encoded string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image string with data URI prefix
    """
    if not os.path.exists(image_path):
        return None
        
    with open(image_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
    # Determine MIME type
    ext = image_path.rsplit('.', 1)[1].lower()
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    return f"data:{mime_type};base64,{encoded}"


def base64_to_image(base64_string: str, save_path: str) -> bool:
    """
    Convert base64 encoded string back to image file.
    
    Args:
        base64_string: Base64 encoded image string
        save_path: Path to save the decoded image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        image_data = base64.b64decode(base64_string)
        
        with open(save_path, 'wb') as f:
            f.write(image_data)
            
        return True
    except Exception as e:
        current_app.logger.error(f"Error converting base64 to image: {e}")
        return False


# =============================================================================
# Authorization Decorators
# =============================================================================

def admin_required(fn):
    """
    Decorator to require admin role for endpoint access.
    
    Usage:
        @admin_bp.route('/users')
        @admin_required
        def get_all_users():
            ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        current_user_id = int(get_jwt_identity())
        user = db.session.get(User, current_user_id)
        
        if not user or user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
            
        return fn(*args, **kwargs)
    return wrapper


def active_user_required(fn):
    """
    Decorator to require an active user account.
    
    Usage:
        @users_bp.route('/profile')
        @jwt_required()
        @active_user_required
        def get_profile():
            ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        current_user_id = int(get_jwt_identity())
        user = db.session.get(User, current_user_id)
        
        if not user or not user.is_active:
            return jsonify({'error': 'Account is inactive or deactivated'}), 403
            
        return fn(*args, **kwargs)
    return wrapper


# =============================================================================
# Response Helpers
# =============================================================================

def success_response(data=None, message=None, status_code=200):
    """
    Create a standardized success response.
    
    Args:
        data: Response data (dict or list)
        message: Success message
        status_code: HTTP status code (default 200)
        
    Returns:
        tuple: (response_dict, status_code)
    """
    response = {
        'success': True,
        'status': 'success'
    }
    
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
        
    return jsonify(response), status_code


def error_response(error, details=None, status_code=400):
    """
    Create a standardized error response.
    
    Args:
        error: Error message
        details: Additional error details
        status_code: HTTP status code (default 400)
        
    Returns:
        tuple: (response_dict, status_code)
    """
    response = {
        'success': False,
        'status': 'error',
        'error': error
    }
    
    if details:
        response['details'] = details
        
    return jsonify(response), status_code


def paginated_response(query, page, per_page, serializer=None):
    """
    Create a paginated response from SQLAlchemy query.
    
    Args:
        query: SQLAlchemy query object
        page: Current page number
        per_page: Items per page
        serializer: Function to serialize each item (optional)
        
    Returns:
        dict: Paginated response data
    """
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    items = pagination.items
    if serializer:
        items = [serializer(item) for item in items]
    elif hasattr(items[0] if items else None, 'to_dict'):
        items = [item.to_dict() for item in items]
        
    return {
        'items': items,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'total': pagination.total,
        'pages': pagination.pages,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev
    }


# =============================================================================
# Misc Helpers
# =============================================================================

def generate_hash(text: str) -> str:
    """
    Generate SHA-256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        str: Hexadecimal hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


def get_severity_color(severity: str) -> str:
    """
    Get color code for severity level.
    
    Args:
        severity: Severity level (mild, medium, severe)
        
    Returns:
        str: Hex color code
    """
    colors = {
        'mild': '#22C55E',      # Green
        'medium': '#F59E0B',    # Amber
        'severe': '#EF4444',    # Red
        'healthy': '#10B981'    # Emerald
    }
    return colors.get(severity.lower(), '#6B7280')  # Gray default
