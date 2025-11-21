from fastapi import APIRouter, HTTPException, Request, Depends

router = APIRouter(prefix="/search", tags=["Search"])

def get_recommender(request: Request):
    return request.state.recommender_system

@router.get(
    "/",
    summary="Search for recipes",
    description="""
    Search for recipes using semantic search and popularity-based ranking.
    
    This endpoint performs the following steps:
    1. Semantic search to find recipes matching the query
    2. Re-ranks results based on popularity (node scores in the graph)
    3. Returns a hybrid list of recommended recipes
    
    **Note:** The system must be fully initialized before this endpoint can be used.
    """,
    responses={
        200: {"description": "List of recommended recipes"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
def search_recipes(
    q: str = "",
    recommender_system = Depends(get_recommender)
):
    if not recommender_system:
        raise HTTPException(status_code=503, detail="System initializing")
    
    results = recommender_system.search(q, num_recs=4)
    return results

@router.get(
    "/recommend/{item_name}",
    summary="Get similar items",
    description="""
    Get recommendations for items similar to the specified item.
    
    This endpoint is ideal for 'You might also like' or 'Similar items' sections
    when a user is viewing a specific recipe.
    
    **Parameters:**
    - `item_name`: The name of the item to find similar items for
    
    **Returns:**
    - A dictionary containing the source item and a list of similar items
    """,
    responses={
        200: {"description": "List of similar items"},
        404: {"description": "Item not found in the graph"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
def get_similar_items(
    item_name: str,
    recommender_system = Depends(get_recommender)
):
    if not recommender_system:
        raise HTTPException(status_code=503, detail="System initializing")
        
    recommendations = recommender_system.get_similar_items(item_name, num_recs=4)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="Item not found in graph")
        
    return {"source_item": item_name, "recommendations": recommendations}

@router.get(
    "/trending",
    summary="Get trending items",
    description="""
    Get the currently most popular items based on global popularity metrics.
    
    This endpoint is perfect for displaying trending or popular items on a home page
    or featured section. The popularity is determined by the number of clicks/interactions
    each item has received.
    
    **Parameters:**
    - `limit`: Maximum number of trending items to return (default: 5, max: 50)
    
    **Returns:**
    - A list of trending items ordered by popularity
    """,
    responses={
        200: {"description": "List of trending items"},
        503: {"description": "Service Unavailable - System is initializing"}
    }
)
def get_trending_items(
    limit: int = 5,
    recommender_system = Depends(get_recommender)
):
    if not recommender_system:
        raise HTTPException(status_code=503, detail="System initializing")
    
    return recommender_system.get_trending(limit=limit)