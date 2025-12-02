# Heart Disease Classification API

A FastAPI backend that serves 5 machine learning models for heart disease classification.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### 3. Access API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Available Endpoints

### Health Check
```
GET /health
```

### Prediction Endpoints
- `POST /predict/coronary` - Coronary artery disease
- `POST /predict/heart-attack` - Heart attack risk
- `POST /predict/heart-failure` - Heart failure risk
- `POST /predict/hypertension` - 10-year CHD risk
- `POST /predict/normal-heart` - Normal heart condition
- `POST /predict/all` - Run all models

## Example Usage

### Using curl
```bash
curl -X POST "http://localhost:8000/predict/coronary" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/coronary",
    json={
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
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Prediction: {result['prediction']}")
```

### Using JavaScript (Frontend)
```javascript
const response = await fetch('http://localhost:8000/predict/coronary', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    exang: 0,
    thal: 2,
    restecg: 1,
    slope: 2,
    thalach: 150,
    chol: 250,
    oldpeak: 1.5,
    trestbps: 130,
    cp: 2,
    ca: 0
  })
});

const result = await response.json();
console.log(`Risk Level: ${result.risk_level}`);
```

## Response Format

All prediction endpoints return:
```json
{
  "model_name": "Model Name",
  "prediction": 0 or 1,
  "probability": 0.XX,
  "risk_level": "Low Risk | Mild Risk | Moderate Risk | High Risk"
}
```

## Models

The API serves 5 pre-trained models:
1. **Coronary** - `best_coronary_xgboost_dedicated.pkl`
2. **Heart Attack** - `best_heartattack_model.pkl`
3. **Heart Failure** - `best_heartfailure_model.pkl`
4. **Hypertension** - `hypertension_model_compressed.pkl`
5. **Normal Heart** - `normal_heart.pkl`

All models are loaded automatically on server startup.

## Testing

Run the test script to verify all endpoints:
```bash
python test_api.py
```

## CORS

CORS is enabled for all origins (development mode). For production, update the `allow_origins` in `main.py` to specific domains.
