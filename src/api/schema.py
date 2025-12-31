from pydantic import BaseModel, Field

# --- Input Schema ---


class HeartDiseaseFeatures(BaseModel):
    """
    Defines the input schema for a single prediction request.
    Based on the columns expected by the model.
    """

    age: float = Field(..., description="Age in years.")
    sex: int = Field(..., description="Sex (1 = male; 0 = female).")
    cp: int = Field(..., description="Chest pain type (1-4, mapped to categories).")
    trestbps: float = Field(
        ...,
        description="Resting blood pressure (in mm Hg on admission to the hospital).",
    )
    chol: float = Field(..., description="Serum cholestoral in mg/dl.")
    fbs: int = Field(
        ..., description="Fasting blood sugar > 120 mg/dl " "(1 = true; 0 = false)."
    )
    restecg: int = Field(
        ..., description="Resting electrocardiographic results (0, 1, or 2)."
    )
    thalach: float = Field(..., description="Maximum heart rate achieved.")
    exang: int = Field(..., description="Exercise induced angina " "(1 = yes; 0 = no).")
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise relative to rest."
    )
    slope: int = Field(
        ..., description="The slope of the peak exercise ST " "segment (1, 2, or 3)."
    )
    ca: int = Field(
        ..., description="Number of major vessels (0-3) colored by flourosopy."
    )
    thal: int = Field(
        ...,
        description="Thalium stress test result "
        "(3 = normal; 6 = fixed defect; 7 = reversable defect).",
    )

    # Example for documentation and validation
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63.0,
                    "sex": 1,
                    "cp": 1,
                    "trestbps": 145.0,
                    "chol": 233.0,
                    "fbs": 1,
                    "restecg": 2,
                    "thalach": 150.0,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 3,
                    "ca": 0,
                    "thal": 6,
                }
            ]
        }
    }


# --- Output Schema ---


class PredictionResult(BaseModel):
    """
    Defines the output schema for the prediction response.
    """

    prediction: int = Field(
        ..., description="Predicted heart disease presence (1) or absence (0)."
    )
    probability: float = Field(
        ..., description="Probability score for the positive class (1)."
    )
    message: str = Field(
        ..., description="A friendly message interpreting the prediction."
    )
