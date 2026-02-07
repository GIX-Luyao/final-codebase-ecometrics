"""Flask application entry point for EcoMetrics backend."""

import logging
import sys

from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from routes import register_blueprints


def setup_logging():
    """Configure dual logging to console and file."""
    logger = logging.getLogger('ecometrics')
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create logs directory if needed
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # File handler (append mode)
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    ))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        Config.RAW_DATA_DIR,
        Config.SESSIONS_DIR,
        Config.CALCULATED_DIR,
        Config.LOGS_DIR
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Setup logging
    logger = setup_logging()

    # Ensure directories exist
    ensure_directories()

    # Register blueprints
    register_blueprints(app)

    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "ecometrics-backend"})

    # Error handlers
    @app.errorhandler(400)
    def handle_bad_request(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "BAD_REQUEST",
                "message": str(e.description)
            }
        }), 400

    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": str(e.description)
            }
        }), 404

    @app.errorhandler(500)
    def handle_internal_error(e):
        logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        }), 500

    logger.info("EcoMetrics backend initialized")

    return app


# Create the app instance
app = create_app()

if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
