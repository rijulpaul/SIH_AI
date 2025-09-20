from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pickle
import numpy as np
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Crop Recommendation API",
    description="API for recommending crops based on soil and weather conditions",
    version="1.0.0"
)

# Load the model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('minmaxscalar.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Crop mapping dictionary (from notebook)
CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Pydantic model for input validation
class CropRecommendationInput(BaseModel):
    N: int = Field(..., ge=0, le=200, description="Nitrogen content in soil (0-200)")
    P: int = Field(..., ge=0, le=200, description="Phosphorus content in soil (0-200)")
    K: int = Field(..., ge=0, le=200, description="Potassium content in soil (0-200)")
    temperature: float = Field(..., ge=-50.0, le=60.0, description="Temperature in Celsius (-50 to 60)")
    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage (0-100)")
    ph: int = Field(..., ge=0, le=14, description="Soil pH level (0-14)")
    rainfall: int = Field(..., ge=0, le=1000, description="Rainfall in mm (0-1000)")
    
    @validator('N', 'P', 'K')
    def validate_nutrients(cls, v):
        if v < 0 or v > 200:
            raise ValueError('Nutrient values must be between 0 and 200')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < -50 or v > 60:
            raise ValueError('Temperature must be between -50 and 60 degrees Celsius')
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Humidity must be between 0 and 100 percent')
        return v
    
    @validator('ph')
    def validate_ph(cls, v):
        if v < 0 or v > 14:
            raise ValueError('pH must be between 0 and 14')
        return v
    
    @validator('rainfall')
    def validate_rainfall(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Rainfall must be between 0 and 1000 mm')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "N": 40,
                "P": 50,
                "K": 50,
                "temperature": 40.0,
                "humidity": 20,
                "ph": 100,
                "rainfall": 100
            }
        }

class CropRecommendationOutput(BaseModel):
    crop_number: int
    crop_name: str
    confidence: Optional[float] = None
    message: str

@app.get("/")
async def root():
    return {"message": "Crop Recommendation API", "status": "running", "available_crops": len(CROP_DICT)}

@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/recommend", response_model=CropRecommendationOutput)
async def recommend_crop(input_data: CropRecommendationInput):
    """
    Recommend a crop based on soil and weather conditions.
    
    Args:
        input_data: Soil nutrients (N, P, K), temperature, humidity, pH, and rainfall
        
    Returns:
        Recommended crop with confidence score
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
    
    try:
        # Convert input to numpy array in the correct order: N, P, K, temperature, humidity, ph, rainfall
        features = np.array([
            input_data.N,
            input_data.P, 
            input_data.K,
            input_data.temperature,
            input_data.humidity,
            input_data.ph,
            input_data.rainfall
        ]).reshape(1, -1)
        
        # Scale the features using the fitted scaler
        scaled_features = scaler.transform(features)
        
        # Make prediction
        crop_number = int(model.predict(scaled_features)[0])
        
        # Get crop name
        crop_name = CROP_DICT.get(crop_number, "Unknown Crop")
        
        # Calculate confidence if the model supports it
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probabilities))
        
        message = f"{crop_name} is the best crop to be cultivated for these conditions"
        
        return CropRecommendationOutput(
            crop_number=crop_number,
            crop_name=crop_name,
            confidence=confidence,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Crop recommendation failed: {str(e)}")

@app.get("/model_info")
async def model_info():
    """
    Get information about the loaded model and available crops.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "scaler_type": type(scaler).__name__ if scaler else None,
        "model_params": model.get_params() if hasattr(model, 'get_params') else None,
        "available_crops": CROP_DICT,
        "required_parameters": {
            "N": "Nitrogen content in soil (0-200)",
            "P": "Phosphorus content in soil (0-200)", 
            "K": "Potassium content in soil (0-200)",
            "temperature": "Temperature in Celsius (-50 to 60)",
            "humidity": "Humidity percentage (0-100)",
            "ph": "Soil pH level (0-14)",
            "rainfall": "Rainfall in mm (0-1000)"
        }
    }
    
    return info

@app.get("/crops")
async def get_available_crops():
    """
    Get list of all available crops for recommendation.
    """
    return {"crops": CROP_DICT, "total_crops": len(CROP_DICT)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)