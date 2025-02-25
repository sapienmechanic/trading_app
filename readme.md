# Inspo Trading App (Version 1.0)

## Overview
This project creates a Dash-based trading dashboard inspired by the concept of a one-stop trading tool, focusing on stock price predictions (AAPL, MSFT, NVDA) using LSTM and basic scraping for sub-industry data.

## Current Status
- `app.py`: Works locally, showing predictions and a simple chart (see screenshot).
- `scraper.py`: Runs but fails to scrape GICS from MSCI—using hardcoded data instead.
- Libraries installed in `venv` with Python 3.11.9 (see requirements.txt).

## Challenges and Fixes
- Fixed missing `sklearn` for MinMaxScaler in `app.py`.
- Resolved `ratelimit` decorator errors in `scraper.py`.
- Handled SQLite errors in `scraper.py` for caching.

## Next Steps (Version 2.0)
- Enhance `app.py` with sub-industry RS, short interest, options data, and richer visualizations using standard trading techniques.
- Refine `scraper.py` with APIs or manual GICS data if scraping fails.
- Test and optimize locally before deployment.

## Disclaimer
This project is for educational purposes only and is not financial advice. It uses public data from Yahoo Finance via `yfinance` and is not affiliated with any financial institution or methodology.