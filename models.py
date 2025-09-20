import pickle
import pandas as pd

class ModelLoader:
    def __init__(self):
        self.crop_recommendation_model = None
        self.crop_recommendation_scaler = None
        self.crop_yield_stats = None
        self.crop_yield_function = None
        self.load_models()
    
    def load_models(self):
        try:
            # Load crop recommendation model
            with open('models/crop_recommendation.pkl', 'rb') as f:
                bundle = pickle.load(f)
                self.crop_recommendation_model = bundle["model"]
                self.crop_recommendation_scaler = bundle["scaler"]
            
            # Load crop yield stats and create function
            with open('models/crop_yield_stats.pkl', 'rb') as f:
                self.crop_yield_stats = pickle.load(f)
            
            # Create the yield prediction function
            self.crop_yield_function = self._create_yield_function()
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _create_yield_function(self):
        """Create the yield prediction function"""
        def predict(crop: str, area_ha: float, temperature_c: float = None,
                    method: str = 'iqr', narrow_pct: float = 0.12) -> dict:
            crop = str(crop).lower()
            if self.crop_yield_stats is None:
                raise ValueError("Crop yield statistics not loaded")
            
            if crop not in self.crop_yield_stats.index.str.lower():
                raise ValueError(f"Crop '{crop}' not found. Available: {list(self.crop_yield_stats.index)}")

            # handle case-insensitive crop names
            matched_crop = [c for c in self.crop_yield_stats.index if c.lower() == crop][0]
            s = self.crop_yield_stats.loc[matched_crop]

            # extract percentiles
            f25, f50, f75 = float(s['fert_p25']), float(s['fert_p50']), float(s['fert_p75'])
            p25, p50, p75 = float(s['pest_p25']), float(s['pest_p50']), float(s['pest_p75'])
            y25, y50, y75 = float(s['yield_p25']), float(s['yield_p50']), float(s['yield_p75'])

            if method == 'iqr':
                fert_low_ha, fert_high_ha = f25, f75
                pest_low_ha, pest_high_ha = p25, p75
                y_low_ha, y_high_ha = y25, y75
            elif method == 'median':
                fert_low_ha, fert_high_ha = f50*(1-narrow_pct), f50*(1+narrow_pct)
                pest_low_ha, pest_high_ha = p50*(1-narrow_pct), p50*(1+narrow_pct)
                y_low_ha, y_high_ha = y50*(1-narrow_pct), y50*(1+narrow_pct)
            else:
                raise ValueError("Unknown method. Use 'iqr' or 'median'.")

            return {
                'Crop': matched_crop,
                'Area_ha': round(float(area_ha),2),
                'Temperature_C_used': None if temperature_c is None else round(float(temperature_c),1),
                'Fertilizer_per_ha_range': (round(fert_low_ha,2), round(fert_high_ha,2)),
                'Pesticide_per_ha_range': (round(pest_low_ha,3), round(pest_high_ha,3)),
                'Yield_per_ha_range': (round(y_low_ha,3), round(y_high_ha,3)),
                'Total_Fertilizer_range': (round(fert_low_ha*area_ha,2), round(fert_high_ha*area_ha,2)),
                'Total_Pesticide_range': (round(pest_low_ha*area_ha,3), round(pest_high_ha*area_ha,3)),
                'Total_Yield_range': (round(y_low_ha*area_ha,3), round(y_high_ha*area_ha,3))
            }
        
        return predict
    
    
    def get_crop_recommendation_model(self):
        return self.crop_recommendation_model
    
    def get_crop_recommendation_scaler(self):
        return self.crop_recommendation_scaler
    
    def get_crop_yield_stats(self):
        return self.crop_yield_stats
    
    def get_crop_yield_function(self):
        return self.crop_yield_function

# Global model loader instance
model_loader = ModelLoader()
