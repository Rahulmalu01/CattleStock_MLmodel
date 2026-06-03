from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    cattle_id: str = Field(..., description="Unique cattle identifier")
    temperature: float = Field(..., description="Body temperature in Celsius")
    heart_rate: int = Field(..., description="Heart rate in beats per minute")
    respiratory_rate: int = Field(..., description="Respiratory rate per minute")
    steps: int = Field(..., description="Number of steps taken")
    activity_level: str = Field(..., description="Activity level: normal, low, high")
    activity_duration_minutes: int = Field(..., description="Duration of activity in minutes")
    rumination_duration: int = Field(..., description="Rumination duration in minutes")
    rumination_episodes: int = Field(..., description="Number of rumination episodes")
    feeding_efficiency: float = Field(..., description="Feeding efficiency percentage")
    milk_yield: float = Field(..., description="Milk yield in liters")
    milk_conductivity: float = Field(..., description="Milk conductivity measurement")
    battery_level: int = Field(default=100, description="IoT sensor battery level")
    signal_strength: int = Field(default=100, description="Signal strength percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "cattle_id": "COW_001",
                "temperature": 39.8,
                "heart_rate": 92,
                "respiratory_rate": 32,
                "steps": 1200,
                "activity_level": "low",
                "activity_duration_minutes": 180,
                "rumination_duration": 240,
                "rumination_episodes": 4,
                "feeding_efficiency": 72.5,
                "milk_yield": 18.2,
                "milk_conductivity": 5.1,
                "battery_level": 84,
                "signal_strength": 90
            }
        }


class DiseaseInfo(BaseModel):
    disease: str = Field(..., description="Predicted disease classification")
    confidence: float = Field(..., description="Confidence score 0-100")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    recommendation: str = Field(..., description="Health recommendation")
    contributing_factors: List[str] = Field(..., description="Top factors influencing prediction")
    model_version: str = Field(..., description="Model version used")


class PredictionResponse(BaseModel):
    status: str = Field("success", description="Response status")
    prediction: DiseaseInfo


class HealthCheckResponse(BaseModel):
    status: str = Field("healthy", description="Service health status")


class ModelInfoResponse(BaseModel):
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    features: List[str]
