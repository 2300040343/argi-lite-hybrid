# =============================================================================
# AgriLite-Hybrid Backend
# app.py - Application Factory & Entry Point
# 
# Flask application factory pattern implementation with extension initialization,
# blueprint registration, error handlers, and ML model loading.
# =============================================================================

import os
import logging
from flask import Flask, jsonify
from config import config, get_config
from extensions import db, migrate, jwt, bcrypt, cors, limiter


def create_app(config_name=None):
    """
    Application factory function.
    
    Creates and configures the Flask application with all extensions,
    blueprints, and error handlers.
    
    Args:
        config_name: Configuration environment ('development', 'testing', 'production')
                    Defaults to FLASK_ENV environment variable or 'development'
                    
    Returns:
        Flask: Configured Flask application instance
    """
    # Determine configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    # Create Flask app instance
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Fail fast if production is misconfigured (missing secrets, wildcard CORS, etc.)
    if config_name == 'production':
        config['production'].validate()
    
    # Setup logging
    setup_logging(app)
    
    # Initialize extensions
    init_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Setup database session handling
    setup_database_handlers(app)
    
    # Load ML model at startup (lazy loading for development)
    with app.app_context():
        init_ml_model(app)
    
    app.logger.info(f"AgriLite-Hybrid API started in {config_name} mode")
    
    return app


def setup_logging(app):
    """
    Configure application logging.
    
    Sets up logging format, level, and handlers based on environment.
    """
    log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set Flask app logger level
    app.logger.setLevel(log_level)


def init_extensions(app):
    """
    Initialize Flask extensions with the application instance.
    
    Extensions are created in extensions.py without app context,
    then initialized here with the app instance.
    """
    # Database ORM
    db.init_app(app)
    
    # Database migrations
    migrate.init_app(app, db)
    
    # JWT authentication
    jwt.init_app(app)
    
    # Password hashing
    bcrypt.init_app(app)
    
    # CORS - Cross Origin Resource Sharing
    # Initialize CORS directly with the app for proper preflight handling
    # Restricted to configured origins (CORS_ORIGINS env var) - never wildcard
    # in production, since supports_credentials=True combined with "*" would
    # allow any site to make authenticated requests on a user's behalf.
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": app.config.get('CORS_ORIGINS', ['*'])}},
         supports_credentials=False,
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'])
    
    # Rate limiting
    limiter.init_app(app)
    
    app.logger.info("Flask extensions initialized")


def register_blueprints(app):
    """
    Register all API route blueprints.
    
    Blueprints organize routes by feature/domain for maintainability.
    All API routes are prefixed with '/api'.
    """
    # Import blueprints
    from routes.auth import auth_bp
    from routes.crops import crops_bp
    from routes.predict import predict_bp
    from routes.history import history_bp
    from routes.admin import admin_bp
    
    # Register blueprints with URL prefixes
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(crops_bp, url_prefix='/api/crops')
    app.register_blueprint(predict_bp, url_prefix='/api/predict')
    app.register_blueprint(history_bp, url_prefix='/api/history')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    
    # Health check endpoint at root level
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint for monitoring and load balancers.
        
        Returns:
            JSON response with API status and database connectivity
        """
        db_healthy = False
        db_message = 'unknown'
        
        try:
            # Test database connection
            db.session.execute(db.text('SELECT 1'))
            db.session.commit()
            db_healthy = True
            db_message = 'connected'
        except Exception as e:
            app.logger.error(f"Database health check failed: {e}")
            db.session.rollback()
            db_message = 'error'
        
        return jsonify({
            'status': 'healthy' if db_healthy else 'degraded',
            'message': 'AgriLite-Hybrid API is running',
            'version': '1.0.0',
            'database': db_message,
            'model_loaded': app.config.get('MODEL_LOADED', False)
        }), 200 if db_healthy else 503
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with API information."""
        return jsonify({
            'name': 'AgriLite-Hybrid API',
            'description': 'Multi-Crop Plant Disease Detection System',
            'version': '1.0.0',
            'documentation': '/api/docs',
            'health': '/health'
        })
    
    app.logger.info("Blueprints registered")


def register_error_handlers(app):
    """
    Register global error handlers for common HTTP errors.
    
    Provides consistent JSON error responses across the API.
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'message': 'Authentication required'
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'success': False,
            'error': 'Forbidden',
            'message': 'You do not have permission to access this resource'
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Not Found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'success': False,
            'error': 'Method Not Allowed',
            'message': f'The {error.description} method is not allowed for this endpoint'
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'success': False,
            'error': 'File Too Large',
            'message': 'The uploaded file exceeds the maximum allowed size'
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'success': False,
            'error': 'Rate Limit Exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        db.session.rollback()
        app.logger.error(f"Internal server error: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500
    
    # JWT Error Handlers
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'success': False,
            'error': 'Token Expired',
            'message': 'Your session has expired. Please login again.'
        }), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'success': False,
            'error': 'Invalid Token',
            'message': 'Token verification failed'
        }), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'success': False,
            'error': 'Authorization Required',
            'message': 'Missing access token'
        }), 401
    
    app.logger.info("Error handlers registered")


def setup_database_handlers(app):
    """
    Setup database session handling for request lifecycle.
    
    Ensures proper cleanup of database sessions after each request
    and handles rollback on exceptions.
    """
    
    @app.after_request
    def after_request(response):
        """Clean up database session after each request."""
        try:
            db.session.remove()
        except Exception as e:
            app.logger.debug(f"Session cleanup notice: {e}")
        return response
    
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Remove database session at the end of request context."""
        if exception:
            db.session.rollback()
        db.session.remove()


def init_ml_model(app):
    """
    Initialize the ML model at application startup.
    
    Loads the trained model into memory for fast inference.
    Uses lazy loading in development to speed up startup.
    """
    try:
        from services.ml_service import MLService
        
        # Initialize ML service
        ml_service = MLService()
        
        # Store in app config for access in routes
        app.config['ML_SERVICE'] = ml_service
        app.config['MODEL_LOADED'] = ml_service.is_loaded()
        
        if ml_service.is_loaded():
            app.logger.info("ML model loaded successfully")
        else:
            app.logger.warning("ML model not loaded - using placeholder predictions")
            
    except Exception as e:
        app.logger.error(f"Error initializing ML service: {e}")
        app.config['ML_SERVICE'] = None
        app.config['MODEL_LOADED'] = False


def create_tables():
    """
    Create all database tables.
    
    Utility function for initial setup or testing.
    """
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == '__main__':
    app = create_app()
    
    # Get port from environment or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    # Run the development server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG']
    )
