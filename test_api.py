#!/usr/bin/env python3
"""
Test script for the Crop Recommendation API
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing Crop Recommendation API...")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return
    
    # Test 2: Get available crops
    try:
        response = requests.get(f"{base_url}/crops")
        print(f"✓ Available crops: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Total crops: {data['total_crops']}")
    except Exception as e:
        print(f"✗ Available crops failed: {e}")
    
    # Test 3: Model info
    try:
        response = requests.get(f"{base_url}/model_info")
        print(f"✓ Model info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Model type: {data['model_type']}")
            print(f"  Scaler type: {data['scaler_type']}")
    except Exception as e:
        print(f"✗ Model info failed: {e}")
    
    # Test 4: Crop recommendation with sample data from notebook
    sample_data = {
        "N": 40,
        "P": 50,
        "K": 50,
        "temperature": 40.0,
        "humidity": 20,
        "ph": 100,
        "rainfall": 100
    }
    
    try:
        response = requests.post(f"{base_url}/recommend", json=sample_data)
        print(f"✓ Crop recommendation: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Recommended crop: {data['crop_name']} (ID: {data['crop_number']})")
            print(f"  Confidence: {data['confidence']}")
            print(f"  Message: {data['message']}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"✗ Crop recommendation failed: {e}")
    
    # Test 5: Test validation with invalid data
    invalid_data = {
        "N": 300,  # Invalid: > 200
        "P": 50,
        "K": 50,
        "temperature": 40.0,
        "humidity": 20,
        "ph": 100,
        "rainfall": 100
    }
    
    try:
        response = requests.post(f"{base_url}/recommend", json=invalid_data)
        print(f"✓ Validation test: {response.status_code}")
        if response.status_code == 422:
            print("  ✓ Correctly rejected invalid data")
        else:
            print(f"  ✗ Should have rejected invalid data: {response.text}")
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
    
    # Test 6: Test missing parameter
    incomplete_data = {
        "N": 40,
        "P": 50,
        "K": 50,
        "temperature": 40.0,
        "humidity": 20,
        "ph": 100
        # Missing rainfall
    }
    
    try:
        response = requests.post(f"{base_url}/recommend", json=incomplete_data)
        print(f"✓ Missing parameter test: {response.status_code}")
        if response.status_code == 422:
            print("  ✓ Correctly rejected incomplete data")
        else:
            print(f"  ✗ Should have rejected incomplete data: {response.text}")
    except Exception as e:
        print(f"✗ Missing parameter test failed: {e}")
    
    print("=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_api()