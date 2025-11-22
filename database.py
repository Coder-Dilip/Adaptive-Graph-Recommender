import sqlite3
import logging
import pandas as pd
from pathlib import Path

DB_NAME = "papers.db"

# --- Database Functions ---

def init_db():
    """Initialize the database with the AI papers table and load data from CSV if empty."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create papers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT,
            pdf_url TEXT,
            primary_category TEXT,
            published TEXT,
            summary TEXT,
            popularity INTEGER DEFAULT 0,
            UNIQUE(title, authors)
        )
    ''')
    
    # Check if empty, if so, load from CSV
    cursor.execute('SELECT count(*) FROM papers')
    if cursor.fetchone()[0] == 0:
        print("🌱 Loading AI papers from CSV...")
        csv_path = Path(__file__).parent / 'ai_papers.csv'
        try:
            df = pd.read_csv(csv_path)
            # Clean and prepare data
            df = df.where(pd.notnull(df), None)  # Convert NaN to None for SQLite
            
            # Insert data into the database, ignoring duplicates
            count_before = cursor.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
            
            # Use to_sql with if_exists='append' and then clean up duplicates
            df.to_sql('temp_papers', conn, if_exists='replace', index=False)
            
            # Insert only new papers that don't already exist
            cursor.execute('''
                INSERT OR IGNORE INTO papers (title, authors, pdf_url, primary_category, published, summary)
                SELECT title, authors, pdf_url, primary_category, published, summary 
                FROM temp_papers
            ''')
            
            # Count how many new papers were added
            count_after = cursor.execute('SELECT COUNT(*) FROM papers').fetchone()[0]
            
            # Clean up
            cursor.execute('DROP TABLE IF EXISTS temp_papers')
            
            print(f"✅ Loaded {count_after - count_before} new papers into the database (skipped {len(df) - (count_after - count_before)} duplicates)")
        except Exception as e:
            print(f"❌ Error loading papers from CSV: {e}")
    
    conn.commit()
    conn.close()

def get_all_papers():
    """Fetch all papers to load into the Recommender RAM."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Allow dict-like access
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, authors, pdf_url, primary_category, 
               published, summary, popularity 
        FROM papers
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def increment_popularity(paper_id: int):
    """Increment the popularity counter for a paper."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE papers SET popularity = popularity + 1 WHERE id = ?",
        (paper_id,)
    )

    if cursor.rowcount == 0:
        logging.warning(f"No paper found with ID: {paper_id}")
    conn.commit()
    conn.close()