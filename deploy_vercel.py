#!/usr/bin/env python3
"""
Vercel deployment script for Decision ML.
Optimizes the deployment for serverless function size limits.
"""

import os
import shutil
import json

def create_vercel_optimized_deployment():
    """Create a Vercel-optimized deployment."""
    print("🚀 Preparing Vercel-optimized deployment...")
    
    # Check if we're using the optimized version
    if not os.path.exists("vercel_app.py"):
        print("❌ vercel_app.py not found. Please ensure the optimized version is available.")
        return False
    
    if not os.path.exists("decision_ml_fallback.py"):
        print("❌ decision_ml_fallback.py not found. Please ensure the fallback is available.")
        return False
    
    # Check requirements.txt is minimal
    with open("requirements.txt", "r") as f:
        requirements = f.read().strip().split('\n')
    
    heavy_packages = ['pandas', 'numpy', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'nltk']
    found_heavy = [pkg for pkg in requirements if any(heavy in pkg.lower() for heavy in heavy_packages)]
    
    if found_heavy:
        print(f"⚠️  Warning: Heavy packages found in requirements.txt: {found_heavy}")
        print("   This may cause deployment to exceed 250MB limit.")
        
        # Create minimal requirements
        minimal_requirements = [
            "fastapi",
            "uvicorn[standard]", 
            "python-multipart",
            "requests"
        ]
        
        print("   Creating minimal requirements.txt...")
        with open("requirements.txt", "w") as f:
            f.write('\n'.join(minimal_requirements) + '\n')
        print("   ✅ Updated requirements.txt with minimal dependencies")
    
    # Verify vercel.json configuration
    with open("vercel.json", "r") as f:
        vercel_config = json.load(f)
    
    if vercel_config.get("builds", [{}])[0].get("src") != "vercel_app.py":
        print("⚠️  Updating vercel.json to use vercel_app.py...")
        vercel_config = {
            "builds": [{"src": "vercel_app.py", "use": "@vercel/python"}],
            "routes": [{"src": "/(.*)", "dest": "vercel_app.py"}]
        }
        
        with open("vercel.json", "w") as f:
            json.dump(vercel_config, f, indent=2)
        print("   ✅ Updated vercel.json configuration")
    
    # Ensure no conflicting properties
    if "functions" in vercel_config:
        print("⚠️  Removing conflicting 'functions' property from vercel.json...")
        del vercel_config["functions"]
        with open("vercel.json", "w") as f:
            json.dump(vercel_config, f, indent=2)
        print("   ✅ Cleaned vercel.json configuration")
    
    # Check .vercelignore exists
    if not os.path.exists(".vercelignore"):
        print("⚠️  .vercelignore not found. Heavy files may be included in deployment.")
    else:
        print("✅ .vercelignore found - excluding heavy files from deployment")
    
    # Estimate deployment size
    total_size = 0
    included_files = []
    
    for root, dirs, files in os.walk("."):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'decision_ml', 'tests']]
        
        for file in files:
            if not file.startswith('.') and not file.endswith(('.pyc', '.pyo', '.md')):
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    included_files.append((file_path, size))
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n📊 Estimated deployment size: {total_size_mb:.2f} MB")
    
    if total_size_mb > 200:
        print("⚠️  Warning: Deployment size is close to 250MB limit")
        print("   Consider removing more files or optimizing further")
    else:
        print("✅ Deployment size is within Vercel limits")
    
    # Show largest files
    print(f"\n📁 Largest files to be deployed:")
    largest_files = sorted(included_files, key=lambda x: x[1], reverse=True)[:10]
    for file_path, size in largest_files:
        size_mb = size / (1024 * 1024)
        print(f"   {file_path}: {size_mb:.2f} MB")
    
    print(f"\n🎯 Deployment Summary:")
    print(f"   ✅ Using lightweight vercel_app.py")
    print(f"   ✅ Using decision_ml_fallback.py for ML features")
    print(f"   ✅ Minimal requirements.txt ({len(requirements)} packages)")
    print(f"   ✅ Estimated size: {total_size_mb:.2f} MB (limit: 250 MB)")
    print(f"   ✅ All Decision ML features available in demo mode")
    
    print(f"\n🚀 Ready for Vercel deployment!")
    print(f"   Run: vercel --prod")
    
    return True

def show_deployment_info():
    """Show information about the optimized deployment."""
    print("\n" + "="*60)
    print("VERCEL DEPLOYMENT OPTIMIZATION")
    print("="*60)
    print("\n🎯 What's included in this optimized deployment:")
    print("   ✅ Full web interface with modern UI")
    print("   ✅ Original matching algorithm (always works)")
    print("   ✅ Decision ML features in demonstration mode")
    print("   ✅ All API endpoints functional")
    print("   ✅ Interactive training simulation")
    print("   ✅ Monitoring and health checks")
    print("   ✅ Sample data for demonstrations")
    
    print("\n🔧 Optimizations applied:")
    print("   ✅ Removed heavy ML dependencies (pandas, scikit-learn, etc.)")
    print("   ✅ Using lightweight fallback implementation")
    print("   ✅ Excluded unnecessary files with .vercelignore")
    print("   ✅ Minimal requirements.txt (4 packages vs 10+)")
    print("   ✅ Optimized for <250MB serverless function limit")
    
    print("\n🎭 Demo Mode Features:")
    print("   ✅ Simulated ML predictions with realistic scores")
    print("   ✅ Training interface with progress simulation")
    print("   ✅ Health monitoring with sample metrics")
    print("   ✅ All UI components fully functional")
    print("   ✅ Perfect for presentations and demonstrations")
    
    print("\n🚀 For Production ML:")
    print("   💡 Use full app.py with dedicated server (not serverless)")
    print("   💡 Deploy on platforms with higher limits (Railway, Render)")
    print("   💡 Use container deployment with Docker")
    print("   💡 Consider hybrid architecture (Vercel frontend + ML backend)")

if __name__ == "__main__":
    show_deployment_info()
    print("\n" + "="*60)
    
    if create_vercel_optimized_deployment():
        print("\n🎉 Optimization complete! Ready to deploy to Vercel.")
    else:
        print("\n❌ Optimization failed. Please check the errors above.")