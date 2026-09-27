"""
WSGI Production Entrypoint for AI Data Analyst Agent.
Suitable for serving with Gunicorn (Linux/macOS), Waitress (Windows/Cross-Platform), or uWSGI.

Usage:
    # On Windows:
    waitress-serve --host=127.0.0.1 --port=5000 wsgi:app

    # On Linux/macOS:
    gunicorn -w 4 -b 127.0.0.1:5000 wsgi:app
"""

import os
from config.settings import Config
from app import create_app

# Create application factory instance
app = create_app()

if __name__ == "__main__":
    # If executed directly, run standard WSGI server
    from waitress import serve
    print(f"Starting production WSGI server with Waitress on http://{Config.HOST}:{Config.PORT}")
    serve(app, host=Config.HOST, port=Config.PORT)
