from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from recommender_engine import RobustRecommender
from datetime import datetime
from fastapi.concurrency import run_in_threadpool
from dependencies.recommender import get_recommender
router = APIRouter(prefix="/api/papers", tags=["Search"])

class PaperResponse(BaseModel):
    id: str
    title: str
    authors: str
    summary: str
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    score:float
    popularity: int = 0

class SearchResponse(BaseModel):
    query: str
    results: List[PaperResponse]
    timestamp: str



@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search for AI research papers",
    description="""
    Search for AI research papers using semantic search and hybrid ranking.
    
    This endpoint performs the following steps:
    1. If query is empty: Returns trending/popular papers
    2. If query provided: Performs semantic search and re-ranks results
    
    Re-ranking considers:
    - Semantic similarity to the query
    - Paper popularity
    - Publication recency
    
    **Note:** The system must be fully initialized before this endpoint can be used.
    """,
    responses={
        200: {"description": "List of recommended papers"},
        400: {"description": "Invalid request parameters"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def search_papers(
    q: str = Query("", min_length=0, description="Search query (empty for popular papers)"),
    current_paper_id = Query(None, description = "Current paper user is viewing while searching"),
    limit: int = Query(5, ge=1, le=20, description="Number of results to return (1-20)"),
    recommender_system: RobustRecommender = Depends(get_recommender)
) -> SearchResponse:
    if not q.strip():
        # Return trending/popular papers for empty query
        results = await run_in_threadpool(recommender_system.get_trending, limit)
    else:
        # Perform normal search for non-empty query
        results = await run_in_threadpool(recommender_system.search, q, current_paper_id, num_recs=limit)
    
    return SearchResponse(
        query=q,
        results=[PaperResponse(**paper) for paper in results],
        timestamp=datetime.utcnow().isoformat()
    )

@router.get(
    "/{paper_id}/similar",
    response_model=List[PaperResponse],
    summary="Get similar papers",
    description="""
    Get recommendations for papers similar to the specified paper.
    
    This endpoint is ideal for 'Related Papers' or 'You might also like' sections
    when a user is viewing a specific paper.
    
    **Parameters:**
    - `paper_id`: The ID of the paper to find similar papers for
    
    **Returns:**
    - A list of similar papers
    """,
    responses={
        200: {"description": "List of similar papers"},
        404: {"description": "Paper not found"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def get_similar_papers(
    paper_id: str,
    limit: int = Query(5, ge=1, le=10, description="Number of similar papers to return (1-10)"),
    recommender_system: RobustRecommender = Depends(get_recommender)
) -> List[PaperResponse]:
    similar = recommender_system.get_similar_items(paper_id, num_recs=limit)
    if not similar:
        raise HTTPException(
            status_code=404,
            detail=f"Paper with ID '{paper_id}' not found or no similar papers available"
        )
    return [PaperResponse(**paper) for paper in similar]

@router.get(
    "/trending",
    response_model=List[PaperResponse],
    summary="Get trending papers",
    description="""
    Get currently trending papers based on popularity and recency.
    
    This endpoint returns papers that are currently popular in the system,
    with a bias towards recently published papers. The ranking is based on:
    - Number of views/clicks (popularity)
    - Publication date (recency)
    
    **Parameters:**
    - `limit`: Maximum number of trending papers to return (1-20, default: 5)
    
    **Returns:**
    - A list of trending papers with their metadata
    """,
    responses={
        200: {"description": "List of trending papers"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
async def get_trending_papers(
    limit: int = Query(5, ge=1, le=20, description="Number of trending papers to return (1-20)"),
    recommender_system: RobustRecommender = Depends(get_recommender)
) -> List[PaperResponse]:
    trending = recommender_system.get_trending(limit)
    return [PaperResponse(**paper) for paper in trending]