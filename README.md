# ML Model Prediction API

A FastAPI backend service that serves machine learning predictions using pre-trained models stored as pickle files.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Model Loading**: Automatically loads pre-trained model and scaler from pickle files
- **Input Validation**: Uses Pydantic for robust input validation
- **Health Checks**: Built-in health check endpoints
- **Model Information**: Endpoint to inspect loaded model details
- **Error Handling**: Comprehensive error handling and status codes

## Project Structure

```
SIH_Model/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── model.pkl           # Trained ML model
├── minmaxscalar.pkl    # MinMaxScaler for feature normalization
└── README.md           # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Alternative: Using Uvicorn Directly

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: Basic API information
- **Response**: 
```json
{
  "message": "ML Model Prediction API",
  "status": "running"
}
```

### 2. Health Check
- **URL**: `GET /health`
- **Description**: Check if the model and scaler are loaded correctly
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Make Prediction
- **URL**: `POST /predict`
- **Description**: Make a prediction using the trained model
- **Request Body**:
```json
{
  "features": [1.0, 2.0, 3.0, 4.0, 5.0]
}
```
- **Response**:
```json
{
  "prediction": 0.85,
  "confidence": 0.92
}
```

### 4. Model Information
- **URL**: `GET /model_info`
- **Description**: Get information about the loaded model
- **Response**:
```json
{
  "model_type": "RandomForestClassifier",
  "scaler_type": "MinMaxScaler",
  "model_params": {...}
}
```

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'

# Get model info
curl http://localhost:8000/model_info
```

### Using Python requests

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Error Handling

The API includes comprehensive error handling:

- **503 Service Unavailable**: When model or scaler fails to load
- **400 Bad Request**: When prediction fails due to invalid input
- **422 Unprocessable Entity**: When input validation fails

## Model Requirements

The API expects:
- `model.pkl`: A trained scikit-learn model
- `minmaxscalar.pkl`: A fitted MinMaxScaler for feature normalization

Both files should be in the same directory as `main.py`.

## Development

### Adding New Endpoints

To add new endpoints, modify `main.py`:

```python
@app.get("/new_endpoint")
async def new_endpoint():
    return {"message": "New endpoint"}
```

### Customizing Input Validation

Modify the `PredictionInput` class to match your model's requirements:

```python
class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: int
    # Add more fields as needed
```

## Deployment

For production deployment, consider:

1. **Environment Variables**: Use environment variables for configuration
2. **Docker**: Containerize the application
3. **Reverse Proxy**: Use nginx or similar for production
4. **Monitoring**: Add logging and monitoring
5. **Security**: Implement authentication and rate limiting

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `model.pkl` and `minmaxscalar.pkl` exist and are valid
2. **Port Already in Use**: Change the port in `main.py` or kill the process using the port
3. **Import Errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`

### Logs

The application logs important events to the console. Check the output for:
- Model loading status
- Request processing
- Error messages