from fastapi import APIRouter, Request, Depends

router = APIRouter(tags=["Health"])

def get_recommender(request: Request):
    return request.state.recommender_system

@router.get(
    "/",
    summary="Health check endpoint",
    description="""
    Check the health and status of the recommendation system.
    
    This endpoint provides information about the current status of the service,
    including whether the recommendation system is fully initialized and ready
    to handle requests.
    
    **Returns:**
    - `status`: Current status of the service ("active" when running)
    - `system_ready`: Boolean indicating if the recommender system is initialized
    - `routes`: List of available API routes
    """,
    responses={
        200: {
            "description": "Service status information",
            "content": {
                "application/json": {
                    "example": {
                        "status": "active",
                        "system_ready": True,
                        "routes": ["/search", "/track/click", "/admin/reset"]
                    }
                }
            }
        }
    }
)
def health_check(recommender_system = Depends(get_recommender)):
    return {
        "status": "active",
        "system_ready": recommender_system is not None,
        "routes": [
            "/search",
            "/track/click",
            "/admin/reset"
        ]
    }
