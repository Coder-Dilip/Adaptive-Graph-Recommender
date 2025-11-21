import sqlite3

DB_NAME = "recipes.db"
# --- Database Functions ---

def init_db():
    """Initialize the database with a table and some dummy data if empty."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            popularity INTEGER DEFAULT 0
        )
    ''')
    
    # Check if empty, if so, seed data
    cursor.execute('SELECT count(*) FROM recipes')
    if cursor.fetchone()[0] == 0:
        print("🌱 Seeding database with initial data...")
        seed_data = [
            ("Veg Burger", "A delicious vegetarian burger with lettuce, tomato, and cheese."),
            ("Chicken Burger", "Juicy chicken patty with mayo, lettuce, and pickles."),
            ("Veg Salad", "Fresh mixed greens with cucumber, tomato, and carrot."),
            ("Chicken Salad", "Grilled chicken on a bed of fresh salad leaves with vinaigrette."),
            ("Margherita Pizza", "Classic pizza with tomato sauce, mozzarella, and basil."),
            ("Pepperoni Pizza", "Spicy pepperoni with tomato sauce and mozzarella cheese."),
            ("Pasta Alfredo", "Creamy Alfredo sauce over fettuccine pasta with parmesan."),
            ("Pasta Arrabiata", "Spicy tomato sauce with garlic and chili over penne pasta."),
            ("Chocolate Cake", "Rich chocolate layered cake with ganache."),
            ("Vanilla Ice Cream", "Creamy vanilla ice cream made with real vanilla beans."),
            ("Momo", "Steamed dumplings filled with spiced meat or vegetables, served with tomato chutney."),
            ("Chatamari", "Nepali rice crepe topped with minced meat, egg, and fresh coriander."),
            ("Sekuwa", "Grilled meat skewers marinated in Nepali spices, served with beaten rice."),
            ("Yomari", "Steamed dumpling made of rice flour with sweet fillings like chaku (molasses) or khuwa."),
            ("Pani Puri", "Hollow crispy puris filled with spiced potatoes and tangy tamarind water."),
            ("Bara", "Savory lentil pancake made from ground black lentils, served with spicy achar."),
            ("Jeri Swari", "Sweet, syrupy dessert made of deep-fried flour dough in spiral shapes."),
            ("Aloo Tama", "Sour and spicy soup made with bamboo shoots, potatoes, and black-eyed peas."),
            ("Yak Cheese Sandwich", "Toasted bread with melted yak cheese, a Himalayan specialty."),
            ("Laphing", "Spicy and tangy cold mung bean noodle dish, a Tibetan-origin street food.")
        ]
        cursor.executemany('INSERT INTO recipes (name, description, popularity) VALUES (?, ?, 0)', seed_data)
        conn.commit()
        
    conn.close()

def get_all_recipes():
    """Fetch all recipes to load into the Recommender RAM."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Allow dict-like access
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recipes")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def increment_popularity(recipe_name: str):
    """Permanent storage of clicks (The Source of Truth)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE recipes SET popularity = popularity + 1 WHERE name = ?", (recipe_name,))
    conn.commit()
    conn.close()