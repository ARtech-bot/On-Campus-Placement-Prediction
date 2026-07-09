"""
api/index.py - FastAPI app for Vercel's Python serverless runtime.

Vercel deploys this whole module as ONE Vercel Function. Model artifacts
(model.pkl, scaler.pkl, feature_columns.pkl, label_mappings.json) must sit
in this same api/ folder and be committed to git — Vercel has no access to
your local filesystem or Google Drive at build time, only what's in the repo.

Routes are prefixed with /api because vercel.json rewrites /api/(.*) to
this function while preserving the original path.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("placement-api")

# Resolve paths relative to this file, NOT the process's working directory —
# Vercel's working directory at request time is not guaranteed to be this folder.
BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ARTIFACTS["model"] = joblib.load(BASE_DIR / "model.pkl")
        ARTIFACTS["scaler"] = joblib.load(BASE_DIR / "scaler.pkl")
        ARTIFACTS["feature_columns"] = joblib.load(BASE_DIR / "feature_columns.pkl")
        with open(BASE_DIR / "label_mappings.json") as f:
            ARTIFACTS["label_mappings"] = json.load(f)
        logger.info("Model artifacts loaded (%s).", ARTIFACTS["label_mappings"].get("best_model"))
    except FileNotFoundError as e:
        logger.error("Model artifacts not found: %s", e)
        ARTIFACTS["model"] = None
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="Student Placement Prediction API",
    version="1.0.0",
    lifespan=lifespan,
    # Docs URLs must carry the /api prefix too, since that's the only path
    # Vercel's rewrite forwards to this function (see vercel.json).
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentProfile(BaseModel):
    CGPA: float = Field(..., ge=0, le=10)
    Internships: int = Field(..., ge=0, le=20)
    Projects: int = Field(..., ge=0, le=50)
    Workshops_Certifications: int = Field(..., ge=0, le=50, alias="Workshops/Certifications")
    AptitudeTestScore: int = Field(..., ge=0, le=100)
    SoftSkillsRating: float = Field(..., ge=0, le=10)
    ExtracurricularActivities: Literal["Yes", "No"]
    PlacementTraining: Literal["Yes", "No"]
    SSC_Marks: int = Field(..., ge=0, le=100)
    HSC_Marks: int = Field(..., ge=0, le=100)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    placement_status: Literal["Placed", "NotPlaced"]
    placement_probability: float
    model_used: str


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_used: str | None = None


def ensure_model_loaded():
    if ARTIFACTS.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts missing from deployment. Commit model.pkl, "
            "scaler.pkl, feature_columns.pkl and label_mappings.json into api/.",
        )


def profile_to_row(profile: StudentProfile) -> np.ndarray:
    mappings = ARTIFACTS["label_mappings"]["binary"]
    data = profile.model_dump(by_alias=True)
    data["ExtracurricularActivities"] = mappings[data["ExtracurricularActivities"]]
    data["PlacementTraining"] = mappings[data["PlacementTraining"]]
    # Order values exactly as feature_columns.pkl specifies — this must match
    # the column order the scaler/model were fit on.
    ordered = [data[col] for col in ARTIFACTS["feature_columns"]]
    return np.array([ordered], dtype=float)


def predict_row(row: np.ndarray) -> PredictionResponse:
    scaled = ARTIFACTS["scaler"].transform(row)
    prob_placed = float(ARTIFACTS["model"].predict_proba(scaled)[0, 1])
    pred = 1 if prob_placed >= 0.5 else 0
    return PredictionResponse(
        placement_status="Placed" if pred == 1 else "NotPlaced",
        placement_probability=round(prob_placed, 4),
        model_used=ARTIFACTS["label_mappings"].get("best_model", "unknown"),
    )


@app.get("/api/health", response_model=HealthResponse)
def health():
    loaded = ARTIFACTS.get("model") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_used=ARTIFACTS.get("label_mappings", {}).get("best_model") if loaded else None,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(profile: StudentProfile):
    ensure_model_loaded()
    try:
        return predict_row(profile_to_row(profile))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(profiles: list[StudentProfile]):
    ensure_model_loaded()
    if not profiles:
        raise HTTPException(status_code=400, detail="Provide at least one student profile.")
    try:
        results = [predict_row(profile_to_row(p)) for p in profiles]
        return BatchPredictionResponse(results=results)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.get("/api")
def root():
    return {"message": "Student Placement Prediction API. Try /api/health or /api/docs."}