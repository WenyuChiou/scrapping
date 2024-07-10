# Scraping historical data for certain stocks
# The recommended modulus for scraping historical stock data include yfinance, ffn, and FinMind

import yfinance as yf # suitablie for US stock market
from FinMind.data import DataLoader  # worked for futrue prices
import ffn
import pandas as pd
import os

# Create the dict of stock name you want to scrap
stock_data = dict()
stock_data['name'] = ['^SPX']

dir = r"C:\Users\user\Desktop\investment\python\scrapping\scraping data"
# Scraping the stock data using yfinance modulus given start and end data
for number, name in enumerate(stock_data['name']):
    #
    df = yf.Ticker(name).history(start="2020-01-01",end="2024-01-01")
    name = name.replace('^','')
    # df.to_csv(os.path.join(dir,name + '_yf.csv')) # save the scraping data as a csv file

# Scraping future prices via FinMind
dl = DataLoader()
future_data = dl.taiwan_futures_daily(futures_id='TX', start_date='1998-01-01')
future_data = future_data[(future_data.trading_session == "position")] #刪除盤後資料
future_data = future_data[(future_data.settlement_price > 0)] #刪除沒有結算資料
future_data = future_data[future_data['contract_date'] == future_data.groupby('date')['contract_date'].transform('min')] #只要近月資料
future_data.to_csv(os.path.join(dir,'台指近月.csv'))

# Scraping prices data using ffn
prices = ffn.get('^SPX', start='2021-01-01')
# Show report
stats = prices.calc_stats()
stats.display()

