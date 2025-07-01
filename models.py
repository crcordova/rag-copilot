from pydantic import BaseModel
from typing import Optional

class FinancialScore(BaseModel):
    roa: Optional[float]
    roe: Optional[float]
    ebitda_margin: Optional[float]
    net_margin: Optional[float]
    current_ratio: Optional[float]
    debt_to_equity: Optional[float]
    asset_turnover: Optional[float]
    revenue_growth: Optional[float]
    net_income_growth: Optional[float]
    score: float
    interpretation: str
