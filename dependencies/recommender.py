from fastapi import Request

def get_recommender(request: Request):
    return request.state.recommender_system