from pydantic import BaseModel

# --- Pydantic Models (Data Validation) ---
class RecipeModel(BaseModel):
    id: int
    name: str
    description: str
    popularity: int = 0

class ClickRequest(BaseModel):
    query: str
    clicked_item_name: str