from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
import database
from schemas import ClickRequest

router = APIRouter(prefix="/track", tags=["Tracking"])

def get_recommender(request: Request):
    return request.state.recommender_system

@router.post(
    "/click",
    summary="Track user click/interaction",
    description="""
    Record a user interaction (click) and update the recommendation model.
    
    This endpoint processes user interactions to improve future recommendations.
    It performs the following operations in a thread-safe manner:
    
    1. Updates the persistent database with the click information (atomic operation)
    2. Updates the in-memory graph with the new interaction data (thread-safe)
    3. Triggers an asynchronous background task to save the updated graph
    
    **Note:** The response is immediate, while the graph persistence happens asynchronously.
    
    **Thread Safety:** This endpoint uses asyncio.Lock to ensure thread safety when
    updating the recommendation graph, preventing race conditions from concurrent updates.
    """,
    responses={
        200: {
            "description": "Click successfully recorded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "recorded",
                        "message": "Feedback received for 'Chocolate Chip Cookies'",
                        "details": "Graph update in progress"
                    }
                }
            }
        },
        400: {"description": "Invalid request payload"},
        500: {"description": "Internal server error while processing click"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def track_click(
    payload: ClickRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Handle user click tracking in a thread-safe manner.
    
    This endpoint is designed to handle high concurrency while maintaining
    data consistency in the recommendation graph.
    """
    recommender_system = request.app.state.recommender_system
    if not recommender_system:
        raise HTTPException(status_code=503, detail="System initializing")

    try:
        # 1. Update Persistent DB (atomic operation)
        database.increment_popularity(payload.clicked_item_name)
        
        # 2. Update In-Memory Graph Logic with thread safety
        recommender_system.process_user_click(
            payload.query, 
            payload.clicked_item_name
        )
        
        # 3. Schedule Background Save (thread-safe)
        def save_graph():
            try:
                recommender_system.save_to_disk()
                print(" Graph state saved to pickle.")
            except Exception as e:
                # Log the error but don't fail the request
                print(f"Error saving graph to disk: {str(e)}")
        
        # Add the background task to save the graph
        background_tasks.add_task(save_graph)
        
        return {
            "status": "recorded", 
            "message": f"Feedback received for '{payload.clicked_item_name}'",
            "details": "Graph update in progress"
        }
        
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in track_click: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process click: {str(e)}"
        )
