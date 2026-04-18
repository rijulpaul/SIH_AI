import pandas as pd
from models import model_loader
from schemas import (
    CropRecommendationInput,
    CropRecommendationOutput,
    CropYieldInput,
    CropYieldOutput,
)


class CropRecommendationService:
    def __init__(self):
        self.model = model_loader.get_crop_recommendation_model()
        self.scaler = model_loader.get_crop_recommendation_scaler()
        self.crop_dict = {
            1: "Rice",
            2: "Maize",
            3: "Jute",
            4: "Cotton",
            5: "Coconut",
            6: "Papaya",
            7: "Orange",
            8: "Apple",
            9: "Muskmelon",
            10: "Watermelon",
            11: "Grapes",
            12: "Mango",
            13: "Banana",
            14: "Pomegranate",
            15: "Lentil",
            16: "Blackgram",
            17: "Mungbean",
            18: "Mothbeans",
            19: "Pigeonpeas",
            20: "Kidneybeans",
            21: "Chickpea",
            22: "Coffee",
        }

    def predict(self, input_data: CropRecommendationInput) -> CropRecommendationOutput:
        if not self.model or not self.scaler:
            raise ValueError("Model not loaded")

        features = pd.DataFrame(
            [
                [
                    input_data.N,
                    input_data.P,
                    input_data.K,
                    input_data.temperature,
                    input_data.humidity,
                    input_data.ph,
                    input_data.rainfall,
                ]
            ],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )

        transformed_features = self.scaler.transform(features)
        prediction = self.model.predict(transformed_features)[0]
        probs = self.model.predict_proba(transformed_features)[0]
        crop_probs = dict()
        for i in range(len(probs)):
            if probs[i] > 0.2:
                crop_name = self.crop_dict.get(i, "Unknown")
                crop_probs[crop_name] = probs[i]

        crops_name = [
            k
            for k, _ in sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)
            ][:3]

        return CropRecommendationOutput(crops=crops_name, message="")


class CropYieldService:
    def __init__(self):
        pass

    def predict(self, input_data: CropYieldInput) -> CropYieldOutput:
        stats = model_loader.get_crop_yield_stats()
        function = model_loader.get_crop_yield_function()

        if stats is None or function is None:
            raise ValueError("Model not loaded")

        result = function(
            crop=input_data.crop,
            area_ha=input_data.area_ha,
            temperature_c=input_data.temperature,
            method=input_data.method,
            narrow_pct=input_data.narrow_pct,
        )

        message = (
            f"Yield prediction for {result['Crop']} on {result['Area_ha']} hectares"
        )
        if result["Temperature_C_used"]:
            message += f" at {result['Temperature_C_used']}°C"

        return CropYieldOutput(
            crop=result["Crop"],
            area_ha=result["Area_ha"],
            temperature_c_used=result["Temperature_C_used"],
            fertilizer_per_ha_range=result["Fertilizer_per_ha_range"],
            pesticide_per_ha_range=result["Pesticide_per_ha_range"],
            yield_per_ha_range=result["Yield_per_ha_range"],
            total_fertilizer_range=result["Total_Fertilizer_range"],
            total_pesticide_range=result["Total_Pesticide_range"],
            total_yield_range=result["Total_Yield_range"],
            message=message,
        )
