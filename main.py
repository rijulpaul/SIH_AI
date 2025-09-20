from fastapi import FastAPI
from routers import crop_recommendation, crop_yield

app = FastAPI()

# Include routers
app.include_router(crop_recommendation.router, prefix="/recommend", tags=["Crop Recommendation"])
app.include_router(crop_yield.router, prefix="/yield", tags=["Crop Yield"])

@app.get("/")
async def root():
    return {"message": "FarmAI API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
