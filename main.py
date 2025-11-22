from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import database
from endpoints import api_router
from recommender_engine import RobustRecommender
import uvicorn
from fastapi.responses import JSONResponse

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize DB and load papers
    print("📚 Initializing database...")
    database.init_db()
    
    # 2. Load Papers Data
    print("📖 Loading papers from database...")
    papers = database.get_all_papers()
    if not papers:
        raise RuntimeError("No papers found in the database. Please ensure the CSV file is properly loaded.")
    print(f"✅ Loaded {len(papers)} papers")
    
    # 3. Initialize AI (Loads Model + Graph)
    print("🚀 Starting AI Recommender System...")
    app.state.recommender_system = RobustRecommender(papers)
    print("✨ Recommender system ready!")
    
    yield
    
    # Shutdown: Final Save
    print("\n🛑 Shutting down, saving graph state...")
    app.state.recommender_system.save_to_disk()

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan, 
    title="AI Paper Recommender",
    description="A hybrid recommender system for AI research papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(api_router)

# Make recommender system available in request state
@app.middleware("http")
async def add_recommender_to_request(request: Request, call_next):
    request.state.recommender_system = request.app.state.recommender_system
    response = await call_next(request)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-paper-recommender"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)