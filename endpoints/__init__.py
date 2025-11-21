from fastapi import APIRouter
from .health import router as health_router
from .search import router as search_router
from .tracking import router as tracking_router
from .admin import router as admin_router

# Create a main router to include all other routers
api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(search_router)
api_router.include_router(tracking_router)
api_router.include_router(admin_router)

__all__ = ["api_router"]
