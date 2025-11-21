# FoodAI Recipe Recommender

A FastAPI-based recommendation system for discovering and exploring recipes. The system uses semantic search and popularity-based ranking to provide personalized recipe recommendations.

## 🚀 Features

- **Semantic Search**: Find recipes using natural language queries
- **Personalized Recommendations**: Get similar recipe suggestions
- **Trending Recipes**: Discover what's popular right now
- **RESTful API**: Easy integration with any frontend
- **Interactive Documentation**: Try out the API directly from your browser

## 🛠️ Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd recommender_system
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   The database will be automatically initialized when you first run the application.

5. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

6. **Access the API documentation**
   - Interactive API docs: http://localhost:8000/docs
   - Alternative API docs: http://localhost:8000/redoc

## 🌐 API Endpoints

### 🔍 Search Recipes
- **GET** `/search/` - Search for recipes using natural language
  - Query Parameters:
    - `q`: Search query (e.g., "vegetarian pasta")

### 🔄 Get Similar Recipes
- **GET** `/search/recommend/{item_name}` - Get recipes similar to the specified item
  - Path Parameters:
    - `item_name`: Name of the recipe to find similar items for

### 📈 Get Trending Recipes
- **GET** `/search/trending` - Get currently trending recipes
  - Query Parameters:
    - `limit`: Number of trending items to return (default: 5)

### 🩺 Health Check
- **GET** `/health` - Check if the API is running

## 🧪 Example API Requests

### Search for recipes
```bash
curl -X 'GET' \
  'http://localhost:8000/search/?q=vegetarian' \
  -H 'accept: application/json'
```

### Get similar recipes
```bash
curl -X 'GET' \
  'http://localhost:8000/search/recommend/Veg%20Burger' \
  -H 'accept: application/json'
```

### Get trending recipes
```bash
curl -X 'GET' \
  'http://localhost:8000/search/trending?limit=3' \
  -H 'accept: application/json'
```

## 🏗️ Project Structure

```
recommender_system/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py              # FastAPI application entry point
├── database.py          # Database initialization and operations
├── recommender_engine.py # Recommendation logic
├── recipes.db           # SQLite database (created on first run)
└── endpoints/           # API endpoint definitions
    ├── __init__.py
    ├── health.py        # Health check endpoint
    ├── search.py        # Search and recommendation endpoints
    └── tracking.py      # User interaction tracking
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, please open an issue in the repository.
