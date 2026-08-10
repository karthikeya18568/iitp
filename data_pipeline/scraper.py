from pathlib import Path
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
DATA_DIR.mkdir(exist_ok=True)

CATEGORY_URLS = {
    'Fiction': 'https://books.toscrape.com/catalogue/category/books/fiction_10/index.html',
    'Mystery': 'https://books.toscrape.com/catalogue/category/books/mystery_3/index.html',
    'Young Adult': 'https://books.toscrape.com/catalogue/category/books/young-adult_21/index.html',
}
HEADERS = {'User-Agent': 'Mozilla/5.0 (educational scraping practice)'}
RATING_WORDS = {'One':1, 'Two':2, 'Three':3, 'Four':4, 'Five':5}


def fetch(url, session):
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def scrape_category(category, url, session):
    html = fetch(url, session)
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for article in soup.select('article.product_pod'):
        title_node = article.select_one('h3 a')
        price_node = article.select_one('p.price_color')
        rating_node = article.select_one('p.star-rating')
        availability_node = article.select_one('p.instock.availability')
        rows.append({
            'title': title_node.get('title', '').strip() if title_node else '',
            'price': price_node.get_text(strip=True) if price_node else '',
            'star_rating': next((c for c in rating_node.get('class', []) if c in RATING_WORDS), '') if rating_node else '',
            'availability': ' '.join(availability_node.stripped_strings) if availability_node else '',
            'category': category,
        })
    return rows


def main():
    session = requests.Session()
    rows = []
    for category, url in CATEGORY_URLS.items():
        category_rows = scrape_category(category, url, session)
        rows.extend(category_rows)
        time.sleep(0.2)
    raw = pd.DataFrame(rows)
    if len(raw) < 60 or raw['category'].nunique() < 3:
        raise RuntimeError(f'Scrape acceptance failed: {len(raw)} rows across {raw.category.nunique()} categories')
    path = DATA_DIR / 'raw_books.csv'
    raw.to_csv(path, index=False)
    print(f'Scraped {len(raw)} books across {raw.category.nunique()} categories.')
    print(raw.head(10).to_string(index=False))
    print(f'Saved: {path}')

if __name__ == '__main__':
    main()
