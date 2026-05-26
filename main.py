"""
Main Entry Point for Document Information Retrieval System

Run this file to start the Flask web application:
    python main.py

The application will be available at http://localhost:5000
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.run import app, logger


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Starting Document IRS System")
    logger.info("=" * 80)
    logger.info("Server running at: http://localhost:5000")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
