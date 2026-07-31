"""
Pydantic validation schemas for California Housing inference API.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class HousingFeatures(BaseModel):
    """
    Input feature schema for California Housing price prediction.
    """
    MedInc: float = Field(..., description="Median income in block group", json_schema_extra={"example": 8.3252})
    HouseAge: float = Field(..., description="Median house age in block group", json_schema_extra={"example": 41.0})
    AveRooms: float = Field(..., description="Average number of rooms per household", json_schema_extra={"example": 6.9841})
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", json_schema_extra={"example": 1.0238})
    Population: float = Field(..., description="Block group population", json_schema_extra={"example": 322.0})
    AveOccup: float = Field(..., description="Average number of household members", json_schema_extra={"example": 2.5556})
    Latitude: float = Field(..., description="Block group latitude", json_schema_extra={"example": 37.88})
    Longitude: float = Field(..., description="Block group longitude", json_schema_extra={"example": -122.23})

    def to_dict(self) -> Dict[str, float]:
        return {
            "MedInc": self.MedInc,
            "HouseAge": self.HouseAge,
            "AveRooms": self.AveRooms,
            "AveBedrms": self.AveBedrms,
            "Population": self.Population,
            "AveOccup": self.AveOccup,
            "Latitude": self.Latitude,
            "Longitude": self.Longitude,
        }


class PredictionResponse(BaseModel):
    """
    Response schema for model prediction endpoint.
    """
    prediction: float = Field(..., description="Predicted median house value (in $100,000s)")
    model_version: str = Field(..., description="Serving model version or stage")
    timestamp: str = Field(..., description="ISO timestamp of prediction")
