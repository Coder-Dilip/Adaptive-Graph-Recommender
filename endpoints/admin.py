import os
import time
from typing import Dict
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import FileResponse
import database
from recommender_engine import RobustRecommender

router = APIRouter(prefix="/admin", tags=["Admin"])

def get_recommender(request: Request):
    return request.state.recommender_system

@router.post("/reset", response_model=Dict[str, str])
def reset_graph(
    request: Request,
    current_recommender = Depends(get_recommender)
) -> Dict[str, str]:
    """
    Reset the recommendation graph to its initial state.
    Deletes the saved graph state and reinitializes the recommender system.
    
    Returns:
        Dict[str, str]: Status message
    """
    try:
        # Remove the saved graph state if it exists
        if os.path.exists("graph_state.pkl"):
            os.remove("graph_state.pkl")
        
        # Re-initialize the recommender system
        recipes = database.get_all_recipes()
        new_recommender = RobustRecommender(recipes)
        
        # Update the app state
        request.app.state.recommender_system = new_recommender
        
        return {"status": "Graph reset to Cold Start"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset graph: {str(e)}"
        )
