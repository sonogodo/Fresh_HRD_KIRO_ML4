#!/usr/bin/env python3
"""
Test script for Decision ML API.
Tests all endpoints and validates functionality.
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class DecisionAPITester:
    """Test class for Decision ML API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Components: {data.get('components', {})}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_training(self) -> bool:
        """Test model training endpoint."""
        print("\n🚀 Testing model training...")
        try:
            # Start training
            response = self.session.post(f"{self.base_url}/decision/train")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Training started: {data['message']}")
                
                # Wait for training to complete (with timeout)
                max_wait = 300  # 5 minutes
                wait_time = 0
                
                while wait_time < max_wait:
                    time.sleep(10)
                    wait_time += 10
                    
                    status_response = self.session.get(f"{self.base_url}/decision/status")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status')
                        
                        print(f"   Training status: {status}")
                        
                        if status == 'completed':
                            print(f"✅ Training completed successfully!")
                            print(f"   Best model: {status_data.get('best_model')}")
                            print(f"   Best score: {status_data.get('best_score')}")
                            return True
                        elif status == 'failed':
                            print(f"❌ Training failed: {status_data.get('message')}")
                            return False
                
                print(f"⏰ Training timeout after {max_wait} seconds")
                return False
            else:
                print(f"❌ Training request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return False
    
    def test_prediction(self) -> bool:
        """Test prediction endpoint."""
        print("\n🎯 Testing prediction...")
        try:
            # Test job descriptions
            test_jobs = [
                "Desenvolvedor Python com experiência em Django e SQL",
                "Engenheiro de Dados especialista em Python, SQL e AWS",
                "Analista de Sistemas com conhecimento em Java e Spring",
                "Cientista de Dados com habilidades em Machine Learning e R"
            ]
            
            for i, job_desc in enumerate(test_jobs):
                print(f"\n   Test {i+1}: {job_desc[:50]}...")
                
                response = self.session.post(
                    f"{self.base_url}/decision/predict",
                    data={
                        "job_description": job_desc,
                        "top_k": 3
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get('top_candidatos', [])
                    
                    print(f"   ✅ Found {len(candidates)} candidates")
                    print(f"   Response time: {data.get('response_time_ms', 0):.1f}ms")
                    print(f"   Model used: {data.get('model_used')}")
                    
                    # Show top candidate
                    if candidates:
                        top_candidate = candidates[0]
                        print(f"   Top match: {top_candidate['candidato']} (score: {top_candidate['score']:.3f})")
                    
                else:
                    print(f"   ❌ Prediction failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            
            print("✅ All prediction tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return False
    
    def test_monitoring(self) -> bool:
        """Test monitoring endpoints."""
        print("\n📊 Testing monitoring...")
        try:
            # Test monitoring health
            response = self.session.get(f"{self.base_url}/decision/monitoring/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Monitoring health: {data.get('overall_status')}")
                print(f"   Predictions 24h: {data.get('metrics', {}).get('total_predictions_24h', 0)}")
            else:
                print(f"⚠️  Monitoring health not available: {response.status_code}")
            
            # Test model summary
            response = self.session.get(f"{self.base_url}/decision/models/summary")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model summary available")
                print(f"   Pipeline status: {data.get('pipeline_status')}")
            else:
                print(f"⚠️  Model summary not available: {response.status_code}")
            
            return True
            
        except Exception as e:
            print(f"❌ Monitoring error: {str(e)}")
            return False
    
    def test_original_endpoints(self) -> bool:
        """Test original matching endpoints for compatibility."""
        print("\n🔄 Testing original endpoints compatibility...")
        try:
            # Test single job matching
            response = self.session.post(
                f"{self.base_url}/match_vaga",
                data={"descricao": "Desenvolvedor Python Junior"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Original single match works: {len(data.get('top_candidatos', []))} candidates")
            else:
                print(f"⚠️  Original single match not available: {response.status_code}")
            
            return True
            
        except Exception as e:
            print(f"❌ Original endpoints error: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        print("🧪 Starting Decision ML API Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Training", self.test_training),
            ("Predictions", self.test_prediction),
            ("Monitoring", self.test_monitoring),
            ("Original Endpoints", self.test_original_endpoints)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Results Summary:")
        print("=" * 50)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:20} {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Decision ML API')
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='Base URL of the API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training test (useful if model is already trained)'
    )
    
    args = parser.parse_args()
    
    tester = DecisionAPITester(args.url)
    
    if args.skip_training:
        # Skip training test
        print("⏭️  Skipping training test as requested")
        tests = [
            ("Health Check", tester.test_health_check),
            ("Predictions", tester.test_prediction),
            ("Monitoring", tester.test_monitoring),
            ("Original Endpoints", tester.test_original_endpoints)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results.append((test_name, False))
        
        passed = sum(1 for _, result in results if result)
        print(f"\nOverall: {passed}/{len(results)} tests passed")
        success = passed == len(results)
    else:
        success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()