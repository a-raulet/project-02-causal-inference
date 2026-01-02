from pydantic import BaseModel, Field
from typing import Literal


class CustomerInput(BaseModel):
    """Input schema for a single customer."""
    recency: int = Field(..., ge=1, le=12, description="Months since last purchase (1-12)")
    history: float = Field(..., ge=0, description="Total amount spent historically")
    history_segment: str = Field(..., description="History segment category")
    mens: Literal[0, 1] = Field(..., description="Has purchased mens products (0 or 1)")
    womens: Literal[0, 1] = Field(..., description="Has purchased womens products (0 or 1)")
    newbie: Literal[0, 1] = Field(..., description="Is a new customer (0 or 1)")
    zip_code: Literal["Rural", "Suburban", "Urban"] = Field(..., description="Customer zip code type")
    channel: Literal["Web", "Phone", "Multichannel"] = Field(..., description="Purchase channel")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recency": 5,
                    "history": 200.0,
                    "history_segment": "3) $200 - $350",
                    "mens": 1,
                    "womens": 0,
                    "newbie": 0,
                    "zip_code": "Urban",
                    "channel": "Web"
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for treatment recommendation."""
    cate_mens_email: float = Field(..., description="CATE for Mens E-Mail treatment")
    cate_womens_email: float = Field(..., description="CATE for Womens E-Mail treatment")
    optimal_treatment: str = Field(..., description="Recommended treatment")
    lift_vs_no_email: float = Field(..., description="Expected conversion lift vs no email")


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    customers: list[CustomerInput] = Field(..., description="List of customers")


class BatchOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: list[PredictionOutput] = Field(..., description="List of predictions")
    summary: dict = Field(..., description="Summary statistics")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
