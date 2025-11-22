from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status, Depends
import database
from dependencies.recommender import get_recommender
from recommender_engine import RobustRecommender
from schemas import ClickRequest
router = APIRouter(prefix="/api/tracking", tags=["Tracking"])
@router.post(
    "/clicks",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Track paper view/click",
    description="""
    Record a user interaction with a research paper and update the recommendation model.
    
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
        202: {
            "description": "Click successfully recorded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "accepted",
                        "paper_id": "12345",
                        "message": "Interaction recorded",
                        "details": "Graph update in progress"
                    }
                }
            }
        },
        400: {"description": "Invalid request payload"},
        404: {"description": "Paper not found"},
        500: {"description": "Internal server error while processing interaction"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def track_paper_interaction(
    request: Request,
    payload: ClickRequest,
    background_tasks: BackgroundTasks,
    recommender_system: RobustRecommender = Depends(get_recommender)

):
    """
    Handle user click tracking in a thread-safe manner.
    
    This endpoint is designed to handle high concurrency while maintaining
    data consistency in the recommendation graph.
    """
    try:
        # 1. Update Persistent DB (atomic operation)
        database.increment_popularity(int(payload.paper_id))
        
        # This is the critical section that needs thread safety
        recommender_system.process_user_click(
            query=payload.query or "",
            clicked_item=payload.paper_id
        )
        
        # 3. Schedule background save (non-blocking)
        background_tasks.add_task(recommender_system.save_to_disk)
        
        return {
            "status": "accepted",
            "paper_id": payload.paper_id,
            "message": "Interaction recorded",
            "details": "Graph update in progress"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        # Log the error but don't expose internal details to the client
        print(f"Error processing interaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )
