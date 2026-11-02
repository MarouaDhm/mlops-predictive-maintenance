from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# Load the saved artifact: model + decision threshold + feature order
MODEL_PATH = Path(__file__).parent / "model" / "predictive_maintenance_model.pkl"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")


class SensorReading(BaseModel):
    """Input schema: raw sensor values. Pydantic validates types and ranges."""
    model_config = ConfigDict(extra="forbid")  # reject unknown fields

    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temp: float = Field(ge=250, le=350, description="Air temperature [K]")
    process_temp: float = Field(ge=250, le=400, description="Process temperature [K]")
    rot_speed: float = Field(ge=0, le=10000, description="Rotational speed [rpm]")
    torque: float = Field(ge=0, le=500, description="Torque [Nm]")
    tool_wear: float = Field(ge=0, le=1000, description="Tool wear [min]")


# Any validation error (wrong type, out of range, bad JSON) returns 400 instead of the default 422
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input", "errors": errors},
    )


@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}


@app.post("/predict")
def predict(reading: SensorReading):
    # Rebuild exactly the same features used in training
    row = {
        "type": {"L": 0, "M": 1, "H": 2}[reading.type],
        "air_temp": reading.air_temp,
        "process_temp": reading.process_temp,
        "rot_speed": reading.rot_speed,
        "torque": reading.torque,
        "tool_wear": reading.tool_wear,
        "temp_diff": reading.process_temp - reading.air_temp,
        "power": reading.torque * reading.rot_speed * 2 * np.pi / 60,
        "wear_torque": reading.tool_wear * reading.torque,
    }
    X = pd.DataFrame([row])[features]  # same column order as training

    probability = float(model.predict_proba(X)[0, 1])
    failure = int(probability >= threshold)

    return {
        "prediction": failure,
        "label": "failure" if failure else "no failure",
        "failure_probability": round(probability, 4),
        "threshold": threshold,
    }
