from ratelimit import limits, sleep_and_retry
import random
import time
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define rate limit using limits
CALLS_PER_MINUTE = 60  # Limit to 60 requests per minute (adjust based on website)
PERIOD = 60  # Period in seconds
rate_limit = limits(calls=CALLS_PER_MINUTE, period=PERIOD)

# Use rate_limit as the argument for sleep_and_retry
@sleep_and_retry(rate_limit)
def make_request(url):
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15'
        ]),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        time.sleep(random.uniform(1, 5))  # Random delay to avoid rate limits
        return response
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def scrape_gics():
    url = "https://www.msci.com/gics"  # Example URL—check the actual site or use yfinance for now
    response = make_request(url)
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        gics_data = []
        for table in tables:
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 2:  # Ensure at least two columns
                    sector = cols[0].text.strip()
                    industry = cols[1].text.strip()
                    if sector and industry:  # Ensure non-empty values
                        gics_data.append({'Sector': sector, 'Industry': industry})
        if not gics_data:  # If no valid data, return None
            logging.warning("No valid GICS data found—skipping cache.")
            return None
        df = pd.DataFrame(gics_data)
        logging.info(f"Scraped GICS DataFrame: {df.head()}")
        try:
            cache_data(df, 'gics_data')
            return df
        except Exception as e:
            logging.error(f"Error caching GICS data: {e}")
            return None
    return None  # Return None if request fails

def cache_data(data, table_name):
    conn = sqlite3.connect('financial_data.db')
    try:
        data.to_sql(table_name, conn, if_exists='replace', index=False)
    except Exception as e:
        logging.error(f"SQLite error caching {table_name}: {e}")
    finally:
        conn.close()

def load_cached_data(table_name):
    conn = sqlite3.connect('financial_data.db')
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        logging.error(f"Error loading {table_name} from SQLite: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error
    finally:
        conn.close()

if __name__ == "__main__":
    logging.info("Starting scraping process...")
    gics_df = scrape_gics()
    if gics_df is not None and not gics_df.empty:
        print("GICS Data Scraped:")
        print(gics_df.head())
    else:
        logging.warning("No GICS data scraped or cached—using hardcoded data in app.py.")
    logging.info("Scraping completed.")