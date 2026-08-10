from pathlib import Path
import sqlite3
import pandas as pd

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / 'data' / 'books.db'
CLEAN_PATH = BASE / 'data' / 'clean_books.csv'

SCHEMA = '''
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0,1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
'''


def load_database(clean_path=CLEAN_PATH, db_path=DB_PATH):
    df = pd.read_csv(clean_path)
    if db_path.exists(): db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON')
    conn.executescript(SCHEMA)
    categories = pd.DataFrame({'category_name': sorted(df['category'].unique())})
    for name in categories['category_name']:
        conn.execute('INSERT INTO categories(category_name) VALUES (?)', (name,))
    cat_map = {name: i+1 for i,name in enumerate(categories['category_name'])}
    books = df.copy()
    books['category_id'] = books['category'].map(cat_map)
    books['in_stock'] = books['in_stock'].astype(int)
    books = books[['title','price_gbp','price_inr','rating','in_stock','category_id']]
    books.to_sql('books', conn, if_exists='append', index=False)
    conn.commit()
    return conn

if __name__ == '__main__':
    conn=load_database(); print(pd.read_sql('SELECT COUNT(*) AS books FROM books',conn)); conn.close()
