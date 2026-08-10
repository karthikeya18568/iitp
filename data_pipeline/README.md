# Module 1 — Data Pipeline

## Objective

Build a raw-to-relational pipeline using `requests` + `BeautifulSoup`, clean the scraped book catalogue, convert GBP to INR using the required fixed rate, load a normalized SQLite database, and demonstrate SQL and pandas querying.

## Data source and scope

The live source is `https://books.toscrape.com/`. The scraper uses the first listing page from three categories: Fiction, Mystery, and Young Adult. This yields at least 60 rows when the site is reachable.

The required conversion is **1 GBP = 105.50 INR**. It is a fixed project-defined baseline; no currency API is used.

## Cleaning decisions

- `price` → `price_gbp`: currency text is parsed to float.
- `star_rating` → `rating`: One–Five is mapped to integers 1–5.
- `availability` → `in_stock`: text containing `In stock` becomes `True`; `Out of stock` becomes `False`.
- Numeric parse failures are median-imputed.
- Rows with malformed availability, missing title, or missing category are dropped because those fields cannot be meaningfully imputed as numeric values.
- `price_inr = price_gbp * 105.50` and is rounded to two decimal places.

## Normalized database

`books.db` contains:

- `categories(category_id PK, category_name UNIQUE)`
- `books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)`

Foreign-key enforcement is enabled with SQLite `PRAGMA foreign_keys = ON`.

## Run end to end

From this directory:

```bash
pip install -r requirements.txt
python pipeline.py
```

The pipeline performs scrape → clean → SQLite load → SQL queries in one run.

## SQL requirements covered

`queries.py` executes five queries covering:

1. `SELECT` + `WHERE`
2. `ORDER BY` + `LIMIT`
3. `DISTINCT`
4. `BETWEEN` + `IN`
5. a two-table `JOIN`

It also reads multiple results using `pd.read_sql()` and reproduces the JOIN with `pd.merge()` in memory. The saved transcript is `sql_outputs/query_outputs.md`.

## Offline smoke-test fixture

`data/raw_books_smoke_test.csv` is only a local development fixture used to test cleaning/database/query code in an environment without internet access. It is **not** a replacement for the live scraper. A normal submission run should execute `python pipeline.py`, which regenerates `data/raw_books.csv` directly from Books to Scrape.
