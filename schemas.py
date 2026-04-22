from pydantic import BaseModel, Field
from typing import Optional, Tuple, List

# Crop Recommendation Schemas
class CropRecommendationInput(BaseModel):
    N: int = Field(..., ge=0, le=200, description="Nitrogen content")
    P: int = Field(..., ge=0, le=200, description="Phosphorus content")
    K: int = Field(..., ge=0, le=200, description="Potassium content")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")
    ph: float = Field(..., ge=0, le=14, description="Soil pH level")
    rainfall: int = Field(..., ge=0, le=2500, description="Rainfall in mm")

class CropRecommendationOutput(BaseModel):
    crops: List[str] 
    message: str

# Crop Yield Schemas
class CropYieldInput(BaseModel):
    crop: str = Field(..., description="Crop name")
    area_ha: float = Field(..., gt=0, description="Area in hectares")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    method: str = Field("iqr", description="Prediction method: 'iqr' or 'median'")
    narrow_pct: float = Field(0.12, ge=0, le=1, description="Narrow percentage for median method")

class CropYieldOutput(BaseModel):
    crop: str
    area_ha: float
    temperature_c_used: Optional[float]
    fertilizer_per_ha_range: Tuple[float, float]
    pesticide_per_ha_range: Tuple[float, float]
    yield_per_ha_range: Tuple[float, float]
    total_fertilizer_range: Tuple[float, float]
    total_pesticide_range: Tuple[float, float]
    total_yield_range: Tuple[float, float]
    message: str
