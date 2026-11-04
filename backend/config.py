# =============================================================================
# AgriLite-Hybrid Backend
# config.py - Configuration Management
# 
# Environment-based configuration for development, testing, and production.
# Uses python-dotenv to load environment variables from .env file.
# =============================================================================

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Base configuration class with default settings.
    All other configuration classes inherit from this.
    """
    
    # ==========================================================================
    # Flask Core Settings
    # ==========================================================================
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # ==========================================================================
    # Database Configuration (Supabase PostgreSQL)
    # ==========================================================================
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///agrilite.db'  # Fallback to SQLite for development
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Connection pooling for production performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20,
        'pool_timeout': 30
    }
    
    # ==========================================================================
    # JWT Authentication Configuration
    # ==========================================================================
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=30)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=60)
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'
    
    # ==========================================================================
    # Rate Limiting Configuration
    # ==========================================================================
    RATELIMIT_ENABLED = os.getenv('RATELIMIT_ENABLED', 'True').lower() == 'true'
    RATELIMIT_STORAGE_URL = os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
    RATELIMIT_STRATEGY = os.getenv('RATELIMIT_STRATEGY', 'fixed-window')
    RATELIMIT_DEFAULT = os.getenv('RATELIMIT_DEFAULT', '200 per hour')
    RATELIMIT_HEADERS_ENABLED = True
    
    # ==========================================================================
    # CORS Configuration
    # ==========================================================================
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    CORS_SUPPORTS_CREDENTIALS = True
    
    # ==========================================================================
    # File Upload Configuration
    # ==========================================================================
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # ==========================================================================
    # ML Model Configuration
    # ==========================================================================
    MODEL_PATH = os.getenv(
        'MODEL_PATH',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    )
    MODEL_NAME = os.getenv('MODEL_NAME', 'agrilite_hybrid.keras')
    CLASS_LABELS_FILE = os.getenv('CLASS_LABELS_FILE', 'class_labels.json')
    
    # ==========================================================================
    # Supabase Configuration (Optional - for cloud features)
    # ==========================================================================
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # ==========================================================================
    # Session Configuration
    # ==========================================================================
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    """
    Development configuration with debug mode enabled.
    Uses SQLite database for easy local development.
    """
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Log SQL queries for debugging
    
    # Use SQLite for development if DATABASE_URL not set
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'sqlite:///agrilite_dev.db'
    )
    
    # Relaxed rate limiting for development
    RATELIMIT_DEFAULT = '1000 per hour'


class TestingConfig(Config):
    """
    Testing configuration for automated tests.
    Uses in-memory SQLite database for fast test execution.
    """
    TESTING = True
    DEBUG = True
    
    # In-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable rate limiting during tests
    RATELIMIT_ENABLED = False
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """
    Production configuration with security hardening.
    Requires all secrets to be set via environment variables.
    """
    DEBUG = False
    TESTING = False
    
    # Production requires proper DATABASE_URL
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    
    # Use Redis for rate limiting in production
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    
    # Stricter rate limits for production
    RATELIMIT_DEFAULT = '100 per hour'
    
    # No insecure fallback in production - must be set via environment
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    
    # Wildcard CORS is not allowed in production - must list real origins
    CORS_ORIGINS = [
        o for o in os.getenv('CORS_ORIGINS', '').split(',') if o and o != '*'
    ]
    
    # Shorter-lived access tokens in production to limit stolen-token exposure
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Secure cookie settings for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'

    @classmethod
    def validate(cls):
        """Fail fast if production is started with insecure/missing settings."""
        errors = []
        if not cls.SECRET_KEY:
            errors.append('SECRET_KEY environment variable is required in production')
        if not cls.JWT_SECRET_KEY:
            errors.append('JWT_SECRET_KEY environment variable is required in production')
        if not cls.SQLALCHEMY_DATABASE_URI:
            errors.append('DATABASE_URL environment variable is required in production')
        if not cls.CORS_ORIGINS:
            errors.append(
                'CORS_ORIGINS environment variable is required in production '
                '(comma-separated list of allowed origins, no wildcard "*")'
            )
        if errors:
            raise RuntimeError(
                'Insecure production configuration:\n- ' + '\n- '.join(errors)
            )


# =============================================================================
# Configuration Dictionary
# Maps environment names to configuration classes
# =============================================================================
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """
    Get the appropriate configuration based on FLASK_ENV environment variable.
    
    Returns:
        Config: Configuration class for the current environment
    """
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
