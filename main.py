from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import database
from endpoints import api_router
from recommender_engine import RobustRecommender

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize DB
    database.init_db()
    
    # 2. Load Data
    recipes = database.get_all_recipes()
    
    # 3. Initialize AI (Loads Model + Graph)
    print("🚀 Starting Recommender System...")
    app.state.recommender_system = RobustRecommender(recipes)
    
    yield
    
    # Shutdown: Final Save
    print("🛑 Shutting down, saving graph...")
    await app.state.recommender_system.save_to_disk()

app = FastAPI(lifespan=lifespan, title="FoodAI Recommender")

# Include all API routes
app.include_router(api_router)

# Make recommender system available in request state
@app.middleware("http")
async def add_recommender_to_request(request: Request, call_next):
    request.state.recommender_system = request.app.state.recommender_system
    response = await call_next(request)
    return response