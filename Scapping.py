# Scraping historical data for certain stocks

import yfinance as yf

df = yf.Ticker("^SPX").history(period="max")
print(df)