"""
Routes module for EcoMetrics Backend.
"""
from flask import Flask

from .sources import sources_bp
from .sessions import sessions_bp
from .config import config_bp
from .calculate import calculate_bp
from .analytics import analytics_bp


def register_routes(app: Flask):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(sources_bp, url_prefix="/api")
    app.register_blueprint(sessions_bp, url_prefix="/api")
    app.register_blueprint(config_bp, url_prefix="/api")
    app.register_blueprint(calculate_bp, url_prefix="/api")
    app.register_blueprint(analytics_bp, url_prefix="/api")

    # Health check endpoint
    @app.route("/api/health")
    def health():
        return {
            "service": "ecometrics-backend",
            "status": "healthy",
            "version": "1.0.0"
        }
