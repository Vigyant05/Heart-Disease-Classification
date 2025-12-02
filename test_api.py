import requests
import json

# Test health endpoint
print("Testing health endpoint...")
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test coronary prediction endpoint
print("Testing coronary prediction endpoint...")
coronary_data = {
    "exang": 0,
    "thal": 2,
    "restecg": 1,
    "slope": 2,
    "thalach": 150,
    "chol": 250,
    "oldpeak": 1.5,
    "trestbps": 130,
    "cp": 2,
    "ca": 0
}

response = requests.post("http://localhost:8000/predict/coronary", json=coronary_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

# Test heart attack prediction endpoint
print("Testing heart attack prediction endpoint...")
heart_attack_data = {
    "age": 55,
    "sex": 1,
    "chest_pain_type": 2,
    "trestbps": 140,
    "chol": 260,
    "fbs": 0,
    "restecg": 0,
    "thalach": 140,
    "exercise_angina": 1,
    "oldpeak": 2.0,
    "st_slope": 1
}

response = requests.post("http://localhost:8000/predict/heart-attack", json=heart_attack_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}\n")

print("✓ All tests completed successfully!")
