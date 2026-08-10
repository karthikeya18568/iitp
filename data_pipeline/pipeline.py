from scraper import main as scrape
from cleaner import main as clean
from database import load_database
from queries import run as run_queries

if __name__ == '__main__':
    scrape()
    clean()
    conn=load_database(); conn.close()
    run_queries()
    print('Data pipeline completed end to end.')
