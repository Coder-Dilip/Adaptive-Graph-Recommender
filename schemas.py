from pydantic import BaseModel
from typing import Optional

# --- Pydantic Models (Data Validation) ---
class RecipeModel(BaseModel):
    id: int
    name: str
    description: str
    popularity: int = 0

class ClickRequest(BaseModel):
    """Request model for tracking paper interactions"""
    paper_id: str
    query: Optional[str] = None