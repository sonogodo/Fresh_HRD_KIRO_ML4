#!/usr/bin/env python3
"""
Test script for vercel_app.py to ensure it works correctly before deployment.
"""

import sys
import os
import json
from datetime import datetime

def test_imports():
    """Test that all imports work correctly."""
    print("🔍 Testing imports...")
    try:
        from vercel_app import app
        from decision_ml_fallback import fallback_pipeline
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_fallback_functionality():
    """Test that the fallback implementation works."""
    print("🔍 Testing fallback functionality...")
    try:
        from decision_ml_fallback import fallback_pipeline
        
        # Test prediction
        matches = fallback_pipeline.predict_matches("Python developer with Django experience", 3)
        if not matches or len(matches) == 0:
            print("❌ No matches returned")
            return False
        
        print(f"✅ Prediction test passed: {len(matches)} matches returned")
        
        # Test training status
        status = fallback_pipeline.get_training_status()
        if not status or "status" not in status:
            print("❌ Invalid training status")
            return False
        
        print(f"✅ Training status test passed: {status['status']}")
        
        # Test health status
        health = fallback_pipeline.get_health_status()
        if not health or "overall_status" not in health:
            print("❌ Invalid health status")
            return False
        
        print(f"✅ Health status test passed: {health['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback functionality error: {e}")
        return False

def test_app_endpoints():
    """Test that the FastAPI app can be created."""
    print("🔍 Testing FastAPI app...")
    try:
        from vercel_app import app
        
        # Check that app is a FastAPI instance
        if not hasattr(app, 'routes'):
            print("❌ App is not a valid FastAPI instance")
            return False
        
        # Count routes
        route_count = len(app.routes)
        print(f"✅ FastAPI app created with {route_count} routes")
        
        # Check for essential routes
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        essential_paths = ['/health', '/decision/predict', '/decision/status']
        
        missing_paths = [path for path in essential_paths if path not in route_paths]
        if missing_paths:
            print(f"⚠️  Missing essential paths: {missing_paths}")
        else:
            print("✅ All essential endpoints present")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("🔍 Testing file structure...")
    
    required_files = [
        "vercel_app.py",
        "decision_ml_fallback.py", 
        "requirements.txt",
        "vercel.json",
        "index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Check requirements.txt is minimal
    with open("requirements.txt", "r") as f:
        requirements = f.read().strip().split('\n')
    
    heavy_packages = ['pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'nltk']
    found_heavy = [pkg for pkg in requirements if any(heavy in pkg.lower() for heavy in heavy_packages)]
    
    if found_heavy:
        print(f"⚠️  Heavy packages found in requirements.txt: {found_heavy}")
        print("   This may cause deployment size issues")
    else:
        print(f"✅ Requirements.txt is minimal ({len(requirements)} packages)")
    
    return True

def test_vercel_config():
    """Test that vercel.json is correctly configured."""
    print("🔍 Testing Vercel configuration...")
    
    try:
        with open("vercel.json", "r") as f:
            config = json.load(f)
        
        # Check builds configuration
        if "builds" not in config:
            print("❌ No 'builds' configuration found")
            return False
        
        if config["builds"][0]["src"] != "vercel_app.py":
            print("❌ Builds not pointing to vercel_app.py")
            return False
        
        # Check for conflicting properties
        if "functions" in config:
            print("❌ Conflicting 'functions' property found")
            return False
        
        print("✅ Vercel configuration is correct")
        return True
        
    except Exception as e:
        print(f"❌ Vercel config error: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall result."""
    print("🧪 Running Vercel deployment tests...")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Fallback Functionality", test_fallback_functionality),
        ("FastAPI App", test_app_endpoints),
        ("Vercel Configuration", test_vercel_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for Vercel deployment.")
        print("\n🚀 Next steps:")
        print("   1. Run: vercel --prod")
        print("   2. Test deployed app")
        print("   3. Verify all features work")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)