from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .schemas import (
    CustomerInput,
    PredictionOutput,
    BatchInput,
    BatchOutput,
    HealthResponse
)
from .preprocessing import preprocess_customer, preprocess_batch
from .models import cate_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        cate_models.load_models()
        print("CATE models loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    yield


app = FastAPI(
    title="CATE Recommendation API",
    description="API for recommending optimal email treatment based on Conditional Average Treatment Effects",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        models_loaded=cate_models.is_loaded
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(customer: CustomerInput):
    """
    Predict optimal email treatment for a single customer.

    Returns CATE estimates for Mens and Womens email campaigns,
    along with the recommended treatment.
    """
    if not cate_models.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run notebook 03_causal_ml.ipynb first."
        )

    # Preprocess input
    X = preprocess_customer(customer.model_dump())

    # Predict CATE
    cate_mens, cate_womens = cate_models.predict(X)
    cate_mens = float(cate_mens[0])
    cate_womens = float(cate_womens[0])

    # Determine optimal treatment
    if cate_mens > cate_womens and cate_mens > 0:
        optimal = "Mens E-Mail"
        lift = cate_mens
    elif cate_womens > cate_mens and cate_womens > 0:
        optimal = "Womens E-Mail"
        lift = cate_womens
    else:
        optimal = "No E-Mail"
        lift = 0.0

    return PredictionOutput(
        cate_mens_email=cate_mens,
        cate_womens_email=cate_womens,
        optimal_treatment=optimal,
        lift_vs_no_email=lift
    )


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    """
    Predict optimal email treatment for multiple customers.

    Returns predictions for each customer plus summary statistics.
    """
    if not cate_models.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run notebook 03_causal_ml.ipynb first."
        )

    # Preprocess all customers
    customers_dict = [c.model_dump() for c in batch.customers]
    X = preprocess_batch(customers_dict)

    # Predict CATE
    cate_mens_arr, cate_womens_arr = cate_models.predict(X)

    # Build predictions
    predictions = []
    treatment_counts = {"Mens E-Mail": 0, "Womens E-Mail": 0, "No E-Mail": 0}

    for cate_mens, cate_womens in zip(cate_mens_arr, cate_womens_arr):
        cate_mens = float(cate_mens)
        cate_womens = float(cate_womens)

        if cate_mens > cate_womens and cate_mens > 0:
            optimal = "Mens E-Mail"
            lift = cate_mens
        elif cate_womens > cate_mens and cate_womens > 0:
            optimal = "Womens E-Mail"
            lift = cate_womens
        else:
            optimal = "No E-Mail"
            lift = 0.0

        treatment_counts[optimal] += 1

        predictions.append(PredictionOutput(
            cate_mens_email=cate_mens,
            cate_womens_email=cate_womens,
            optimal_treatment=optimal,
            lift_vs_no_email=lift
        ))

    # Summary
    n = len(predictions)
    summary = {
        "total_customers": n,
        "treatment_distribution": {
            k: {"count": v, "percentage": round(v / n * 100, 1)}
            for k, v in treatment_counts.items()
        },
        "avg_cate_mens": round(sum(p.cate_mens_email for p in predictions) / n, 4),
        "avg_cate_womens": round(sum(p.cate_womens_email for p in predictions) / n, 4)
    }

    return BatchOutput(predictions=predictions, summary=summary)
