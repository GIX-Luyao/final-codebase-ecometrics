"""Routes package for EcoMetrics backend."""

from flask import Blueprint

from .raw_data import raw_data_bp
from .data_update import data_update_bp
from .calculated import calculated_bp


def register_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(raw_data_bp)
    app.register_blueprint(data_update_bp)
    app.register_blueprint(calculated_bp)


__all__ = [
    "raw_data_bp",
    "data_update_bp",
    "calculated_bp",
    "register_blueprints"
]
