from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # Register blueprints
    from .routes import main_routes
    CORS(app)  # Enable CORS for all routes
    app.register_blueprint(main_routes)

    # Any additional configuration can go here
    app.config['UPLOAD_FOLDER'] = 'uploads'

    return app
