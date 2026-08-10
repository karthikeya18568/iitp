from pathlib import Path
import re
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
GBP_TO_INR = 105.50
RATING_MAP = {'One':1, 'Two':2, 'Three':3, 'Four':4, 'Five':5}


def parse_price(value):
    if pd.isna(value): return np.nan
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)', str(value))
    return float(m.group(1)) if m else np.nan


def parse_rating(value):
    if pd.isna(value): return np.nan
    return RATING_MAP.get(str(value).strip(), np.nan)


def parse_stock(value):
    if pd.isna(value): return np.nan
    text = str(value).strip().lower()
    if 'in stock' in text: return True
    if 'out of stock' in text: return False
    return np.nan


def clean_books(raw):
    df = raw.copy()
    df['price_gbp'] = df['price'].map(parse_price)
    df['rating'] = df['star_rating'].map(parse_rating)
    df['in_stock'] = df['availability'].map(parse_stock)

    # Numeric parsing failures use median imputation, as required.
    for col in ['price_gbp', 'rating']:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    # Boolean parsing cannot use numeric median; drop malformed availability rows.
    df = df.dropna(subset=['in_stock', 'title', 'category']).copy()
    df['rating'] = df['rating'].round().astype(int).clip(1,5)
    df['price_gbp'] = df['price_gbp'].astype(float)
    df['in_stock'] = df['in_stock'].astype(bool)
    df['price_inr'] = (df['price_gbp'] * GBP_TO_INR).round(2)
    return df[['title','price_gbp','price_inr','rating','in_stock','category']]


def main():
    raw_path = DATA_DIR / 'raw_books.csv'
    clean_path = DATA_DIR / 'clean_books.csv'
    raw = pd.read_csv(raw_path)
    clean = clean_books(raw)
    if len(clean) < 60 or clean['category'].nunique() < 3:
        raise RuntimeError('Cleaned dataset does not meet the >=60 rows / >=3 categories requirement.')
    clean.to_csv(clean_path, index=False)
    print(f'Cleaned rows: {len(clean)}')
    print(clean.dtypes)
    print(clean.head(10).to_string(index=False))
    print(f'Saved: {clean_path}')

if __name__ == '__main__': main()
