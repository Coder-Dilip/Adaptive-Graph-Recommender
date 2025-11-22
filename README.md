# AI Research Paper Recommender System

A high-performance, hybrid recommender system for AI research papers, powered by FastAPI. This system provides intelligent paper recommendations using a combination of semantic search, collaborative filtering, and graph-based algorithms.

## Features

- **Semantic Search**: Find relevant papers using natural language queries
- **Smart Recommendations**: Get personalized paper recommendations based on user interactions
- **Trending Papers**: Discover popular and trending AI research papers
- **Similar Papers**: Find papers similar to a given paper
- **Interaction Tracking**: Track user interactions to improve recommendations
- **RESTful API**: Fully documented OpenAPI/Swagger interface

## API Endpoints

### Search & Recommendations

#### `GET /api/papers/search`
- **Description**: Search for AI research papers with semantic search and hybrid ranking
- **Query Parameters**:
  - `q`: Search query (optional, returns trending papers if empty)
  - `limit`: Number of results to return (default: 10, max: 20)
- **Response**: List of relevant papers with metadata and relevance scores

#### `GET /api/papers/similar/{paper_id}`
- **Description**: Get papers similar to a specific paper
- **Path Parameters**:
  - `paper_id`: ID of the target paper
- **Query Parameters**:
  - `limit`: Number of similar papers to return (default: 5, max: 10)
- **Response**: List of similar papers with similarity scores

#### `GET /api/papers/trending`
- **Description**: Get currently trending papers based on recent interactions
- **Query Parameters**:
  - `limit`: Number of trending papers to return (default: 5, max: 20)
- **Response**: List of trending papers with popularity metrics

### Tracking

#### `POST /api/tracking/clicks`
- **Description**: Record user interactions with papers
- **Request Body**:
  ```json
  {
    "paper_id": "string",
    "user_id": "string",
    "interaction_type": "view|click|download",
    "timestamp": "2025-11-22T17:30:00Z"
  }
  ```
- **Response**: Confirmation of recorded interaction

### Health Check

#### `GET /health`
- **Description**: Check if the service is running
- **Response**: Service status and version information

## Setup and Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd recommender_system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Development Mode

```bash
uvicorn main:app --reload
```

### Production Mode

For production, use a production ASGI server like uvicorn with gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## API Documentation

Once the application is running, you can access the following:

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Environment Variables

The application can be configured using the following environment variables:

- `DATABASE_URL`: Database connection string (default: SQLite database)
- `MODEL_NAME`: Name of the sentence transformer model (default: 'all-MiniLM-L6-v2')
- `DEBUG`: Enable debug mode (default: False)

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.