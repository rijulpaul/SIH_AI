from fastapi import APIRouter, HTTPException
from schemas import CropYieldInput, CropYieldOutput
from services import CropYieldService

router = APIRouter()
service = CropYieldService()

@router.post("/", response_model=CropYieldOutput)
async def predict_yield(input_data: CropYieldInput):
    try:
        result = service.predict(input_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
