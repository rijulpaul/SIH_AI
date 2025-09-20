from fastapi import APIRouter, HTTPException
from schemas import CropRecommendationInput, CropRecommendationOutput
from services import CropRecommendationService

router = APIRouter()
service = CropRecommendationService()

@router.post("/", response_model=CropRecommendationOutput)
async def recommend_crop(input_data: CropRecommendationInput):
    try:
        result = service.predict(input_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
