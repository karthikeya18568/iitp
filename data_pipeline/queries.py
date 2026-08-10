from pathlib import Path
import sqlite3
import pandas as pd

BASE=Path(__file__).resolve().parent
DB=BASE/'data'/'books.db'
OUT=BASE/'sql_outputs'; OUT.mkdir(exist_ok=True)

QUERIES={
'01_select_where': "SELECT title, price_gbp, rating FROM books WHERE rating >= 4 AND price_gbp < 30 ORDER BY price_gbp;",
'02_order_by_limit': "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
'03_distinct': "SELECT DISTINCT rating FROM books ORDER BY rating;",
'04_between_in': "SELECT title, price_gbp, category_id FROM books WHERE price_gbp BETWEEN 15 AND 30 AND category_id IN (1,2,3) ORDER BY price_gbp;",
'05_join': "SELECT b.title, c.category_name, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;",
}


def run():
    conn=sqlite3.connect(DB)
    outputs=[]
    for name,sql in QUERIES.items():
        df=pd.read_sql(sql,conn)
        df.to_csv(OUT/f'{name}.csv',index=False)
        outputs.append(f'## {name}\n\n```sql\n{sql}\n```\n\n```text\n{df.to_string(index=False)}\n```\n')
    # pd.read_sql for at least two results.
    read_sql_1=pd.read_sql(QUERIES['01_select_where'],conn)
    read_sql_2=pd.read_sql(QUERIES['02_order_by_limit'],conn)
    # In-memory merge reproduction of the JOIN query.
    books=pd.read_sql('SELECT * FROM books',conn)
    cats=pd.read_sql('SELECT * FROM categories',conn)
    merged=books.merge(cats,on='category_id',how='inner')
    merged=merged[['title','category_name','rating','price_inr']].sort_values(['rating','price_inr'],ascending=[False,False]).head(10).reset_index(drop=True)
    sql_join=pd.read_sql(QUERIES['05_join'],conn).reset_index(drop=True)
    equivalent=merged.equals(sql_join)
    outputs.append('## pandas.read_sql and pandas.merge equivalence\n')
    outputs.append('`pd.read_sql_query` result for the JOIN:\n\n```text\n'+sql_join.to_string(index=False)+'\n```\n')
    outputs.append('`pd.merge` reproduction:\n\n```text\n'+merged.to_string(index=False)+'\n```\n')
    outputs.append(f'**Equivalent outputs:** `{equivalent}`\n')
    (OUT/'query_outputs.md').write_text('\n'.join(outputs),encoding='utf-8')
    print((OUT/'query_outputs.md').read_text())
    conn.close()

if __name__=='__main__': run()
