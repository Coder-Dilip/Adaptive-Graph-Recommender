import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import database
from endpoints import api_router
from recommender_engine import RobustRecommender
import uvicorn
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

# --- Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 1. Initialize DB and load papers
        print("\n[1/4] Initializing database...")
        database.init_db()
        
        # 2. Load Papers Data
        print("[2/4] Loading papers from database...")
        papers = database.get_all_papers()
        if not papers:
            raise RuntimeError("No papers found in the database. Please ensure the CSV file is properly loaded.")
        print(f" Loaded {len(papers)} papers")
    except Exception as e:
        print(f" Error during database initialization: {str(e)}")
        raise
    
    try:
        # 3. Initialize AI (Loads Model + Graph)
        print("[3/4] Starting AI Recommender System...")
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not qdrant_url or not qdrant_api_key:
            print("  QDRANT_URL or QDRANT_API_KEY not found in .env file. Using local Qdrant instance.")
        
        print("   - Initializing Qdrant client...")
        print(f"   - Qdrant URL: {qdrant_url if qdrant_url else 'localhost:6333'}")
        
        print("   - Loading sentence transformer model...")
        app.state.recommender_system = RobustRecommender(
            papers,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )
        print(" Recommender system ready!")
    except Exception as e:
        print(f" Error initializing recommender system: {str(e)}")
        raise
    
    yield
    
    # Shutdown: Final Save
    print("\n Shutting down...")

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