#!/usr/bin/env python3
"""
Setup script for Decision ML system.
Handles installation, data preparation, and initial setup.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import argparse

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    requirements_file = "decision_ml/requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"   ❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing packages from requirements.txt"
    )

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    
    nltk_command = """python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Error downloading NLTK data: {e}')
    exit(1)
" """
    
    return run_command(nltk_command, "Downloading NLTK punkt and stopwords")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "decision_ml/models",
        "decision_ml/logs",
        "decision_ml/tests/__pycache__"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"   ✅ Created/verified: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")
            return False
    
    return True

def check_data_files():
    """Check if required data files exist."""
    print("📊 Checking data files...")
    
    required_files = [
        "JSONs_DECISION/vagas_padrao.json",
        "JSONs_DECISION/candidates.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ Found: {file_path}")
            
            # Check file size
            size = os.path.getsize(file_path)
            print(f"      Size: {size:,} bytes")
            
            # Validate JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"      Records: {len(data)} items")
                    elif isinstance(data, list):
                        print(f"      Records: {len(data)} items")
            except Exception as e:
                print(f"      ⚠️  JSON validation error: {e}")
        else:
            print(f"   ❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def run_tests():
    """Run unit tests."""
    print("🧪 Running unit tests...")
    
    test_command = "python -m pytest decision_ml/tests/ -v --tb=short"
    return run_command(test_command, "Running pytest")

def train_initial_model():
    """Train the initial model."""
    print("🚀 Training initial model...")
    
    train_command = "python train_decision_model.py --log-level INFO"
    return run_command(train_command, "Training Decision ML model")

def test_api():
    """Test the API endpoints."""
    print("🌐 Testing API...")
    
    # Start API in background and test
    test_command = "python test_decision_api.py --skip-training"
    return run_command(test_command, "Testing API endpoints")

def setup_docker():
    """Setup Docker configuration."""
    print("🐳 Setting up Docker...")
    
    if not run_command("docker --version", "Checking Docker installation"):
        print("   ⚠️  Docker not found. Skipping Docker setup.")
        return True
    
    # Build Docker image
    if not run_command("docker build -t decision-ml .", "Building Docker image"):
        return False
    
    print("   ✅ Docker image built successfully")
    print("   💡 You can now run: docker run -p 8000:8000 decision-ml")
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Decision ML system')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-training', action='store_true', help='Skip initial model training')
    parser.add_argument('--skip-docker', action='store_true', help='Skip Docker setup')
    parser.add_argument('--quick', action='store_true', help='Quick setup (skip tests, training, and docker)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.skip_tests = True
        args.skip_training = True
        args.skip_docker = True
    
    print("🎯 Decision ML Setup Script")
    print("=" * 50)
    
    # Setup steps
    steps = [
        ("Check Python version", check_python_version),
        ("Create directories", create_directories),
        ("Install dependencies", install_dependencies),
        ("Download NLTK data", download_nltk_data),
        ("Check data files", check_data_files),
    ]
    
    if not args.skip_tests:
        steps.append(("Run tests", run_tests))
    
    if not args.skip_training:
        steps.append(("Train initial model", train_initial_model))
    
    if not args.skip_docker:
        steps.append(("Setup Docker", setup_docker))
    
    # Execute steps
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed")
            else:
                print(f"✅ {step_name} completed")
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    if failed_steps:
        print(f"❌ {len(failed_steps)} step(s) failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\n⚠️  Please fix the issues above and run setup again.")
        
        # Provide troubleshooting tips
        print("\n💡 Troubleshooting tips:")
        if "Install dependencies" in failed_steps:
            print("   - Try: pip install --upgrade pip")
            print("   - Try: pip install -r decision_ml/requirements.txt --no-cache-dir")
        
        if "Check data files" in failed_steps:
            print("   - Ensure JSONs_DECISION/ folder contains the required data files")
            print("   - Check file permissions and encoding")
        
        if "Train initial model" in failed_steps:
            print("   - Check available memory (requires ~2GB)")
            print("   - Try reducing dataset size for testing")
        
        sys.exit(1)
    else:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Start the API: python -m uvicorn decision_ml.api:app --reload")
        print("   2. Or use Docker: docker run -p 8000:8000 decision-ml")
        print("   3. Test the API: python test_decision_api.py")
        print("   4. Visit: http://localhost:8000/docs for API documentation")
        
        sys.exit(0)

if __name__ == '__main__':
    main()