"""
EcoMetrics GHG Emissions Data Engine - Flask Application Entry Point.

A declarative data mapping and emissions calculation engine for Seattle Public Utilities.
"""
import logging
from flask import Flask
from flask_cors import CORS

from config import Config
from routes import register_routes


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(Config)

    # Ensure directories exist
    Config.ensure_directories()

    # Configure CORS
    CORS(app, origins=[
    "http://localhost:3000",
    "https://main.xxxxxxxxx.amplifyapp.com"
    ])

    # Configure logging
    setup_logging()

    # Register routes
    register_routes(app)

    return app


def setup_logging():
    """Configure application logging."""
    # Create custom logging adapter for session context
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


class SessionLogger:
    """Logger with session context."""

    def __init__(self, session_id: str = "-"):
        self.session_id = session_id
        self.logger = logging.getLogger("ecometrics")

    def _format_message(self, component: str, action: str, status: str, message: str) -> str:
        return f"[{self.session_id}] {component} - {action} - {status} - {message}"

    def info(self, component: str, action: str, status: str, message: str):
        self.logger.info(self._format_message(component, action, status, message))

    def error(self, component: str, action: str, status: str, message: str):
        self.logger.error(self._format_message(component, action, status, message))

    def warning(self, component: str, action: str, status: str, message: str):
        self.logger.warning(self._format_message(component, action, status, message))


# Create application instance
app = create_app()


if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
