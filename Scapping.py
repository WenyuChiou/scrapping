# Scraping historical data for certain stocks

import yfinance as yf
import pandas as pd
import os

# Create the dict of stock name you want to scrap
stock_data = dict()
stock_data['name'] = ["2330.TW",'^SPX']

dir = r"C:\Users\user\Desktop\investment\python\scrapping\scraping data"
# Scraping the stock data using yfinance modulus given start and end data
for number, name in enumerate(stock_data['name']):
    df = yf.Ticker(name).history(start="2020-01-01",end="2024-01-01")
    name = name.replace('^','')
    df.to_csv(os.path.join(dir,name + '.csv')) # save the scraping data as a csv file





