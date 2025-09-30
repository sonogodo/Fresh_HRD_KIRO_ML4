#!/usr/bin/env python3
"""
Initialization script for Decision ML components.
This script ensures all necessary components are available for Vercel deployment.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure necessary directories exist."""
    directories = [
        "decision_ml/models",
        "decision_ml/logs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")

def download_nltk_data():
    """Download required NLTK data if available."""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not download NLTK data: {e}")

def check_decision_ml_availability():
    """Check if Decision ML components are available."""
    try:
        from decision_ml.pipeline import DecisionMLPipeline
        from decision_ml.monitoring import monitor
        logger.info("✅ Decision ML components are available")
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Decision ML components not available: {e}")
        return False

def initialize():
    """Initialize Decision ML for deployment."""
    logger.info("🚀 Initializing Decision ML components...")
    
    # Ensure directories
    ensure_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Check availability
    ml_available = check_decision_ml_availability()
    
    if ml_available:
        logger.info("✅ Decision ML initialization completed successfully")
    else:
        logger.warning("⚠️ Decision ML initialization completed with warnings")
    
    return ml_available

if __name__ == '__main__':
    initialize()