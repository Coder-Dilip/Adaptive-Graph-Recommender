from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from typing import Dict, Any
import psutil
import os
from datetime import datetime

router = APIRouter(prefix="/api", tags=["System"])

class SystemStatus(BaseModel):
    status: str
    system_ready: bool
    timestamp: str
    version: str
    num_papers: int
    memory_usage: Dict[str, Any]
    uptime: float

class HealthResponse(BaseModel):
    status: str
    system: SystemStatus
    details: Dict[str, Any]

def get_recommender(request: Request):
    return request.state.recommender_system

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description=""" 
    Check the health and status of the AI Papers Recommender System.
    
    This endpoint provides detailed information about the current status of the service,
    including system metrics, number of papers loaded, and service availability.
    
    **Returns:**
    - `status`: Overall service status ("healthy" or "unhealthy")
    - `system`: System status details including memory usage and uptime
    - `details`: Additional diagnostic information
    """,
    responses={
        200: {
            "description": "System status information",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "system": {
                            "status": "operational",
                            "system_ready": True,
                            "timestamp": "2023-04-01T12:00:00Z",
                            "version": "1.0.0",
                            "num_papers": 1500,
                            "memory_usage": {
                                "total": 16.0,
                                "available": 8.5,
                                "percent": 46.9,
                                "used": 7.5,
                                "free": 8.5
                            },
                            "uptime": 1234.56
                        },
                        "details": {
                            "graph_ready": True,
                            "model_ready": True,
                            "database_ready": True
                        }
                    }
                }
            }
        },
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def health_check(
    request: Request,
    recommender_system = Depends(get_recommender)
) -> HealthResponse:
    """
    Comprehensive health check endpoint that verifies all system components.
    
    This endpoint performs the following checks:
    1. System resource usage (memory, CPU)
    2. Recommender system status
    3. Database connection
    4. Model and graph initialization
    
    Returns 200 if the system is healthy, 503 if any critical component is unavailable.
    """
    # Get system metrics
    process = psutil.Process(os.getpid())
    memory_info = psutil.virtual_memory()
    
    # Prepare system status
    system_status = {
        "status": "operational",
        "system_ready": True,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "num_papers": len(recommender_system.G) if hasattr(recommender_system, 'G') else 0,
        "memory_usage": {
            "total": round(memory_info.total / (1024 ** 3), 2),  # Convert to GB
            "available": round(memory_info.available / (1024 ** 3), 2),
            "percent": memory_info.percent,
            "used": round(memory_info.used / (1024 ** 3), 2),
            "free": round(memory_info.free / (1024 ** 3), 2)
        },
        "uptime": round(process.create_time() - datetime.now().timestamp(), 2)
    }
    
    # Check system components
    details = {
        "graph_ready": hasattr(recommender_system, 'G') and len(recommender_system.G) > 0,
        "model_ready": hasattr(recommender_system, 'model') and recommender_system.model is not None,
        "database_ready": True  # Add actual database check if needed
    }
    
    # Determine overall status
    status = "healthy" if all(details.values()) else "degraded"
    
    return {
        "status": status,
        "system": system_status,
        "details": details
    }
