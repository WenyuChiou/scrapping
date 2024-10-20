#%%
# main_trading_script.py

import os
from scraping_and_indicators import scrape_prices_yfinance, add_specified_indicators


# Directory for saving files
dir = r"C:\Users\user\Desktop\investment\python\scrapping\scraping_data"
os.makedirs(dir, exist_ok=True)

# Step 1: Scrape historical data for a given stock ticker
ticker = 'TSM'
start_date = '2021-01-01'
end_date = '2024-10-18'
prices = scrape_prices_yfinance(ticker, start_date, end_date)

# Step 2: Add specified indicators if data is present
if not prices.empty:
    # Define the indicators to add and custom settings
    indicator_list = ['SMA', 'RSI', 'BBANDS', 'MACD', 'ATR']
    custom_settings = {
        'SMA': {'timeperiod': 50},  # Custom SMA period
        'RSI': {'timeperiod': 14},   # Custom RSI period
        'BBANDS': {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'price': 'close'},  # Custom Bollinger Bands settings
        'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},  # Custom MACD settings
        'ATR': {'timeperiod': 14}  # Custom ATR settings
    }

    # Add indicators using the add_specified_indicators function
    prices_with_indicators = add_specified_indicators(prices, indicator_list, custom_settings)

    # Step 3: Save the updated DataFrame with indicators to a CSV file
    csv_filename = os.path.join(dir, f'{ticker}_with_specified_indicators_yfinance.csv')
    prices_with_indicators.to_csv(csv_filename)
    print(f"Updated data saved to {csv_filename}")
else:
    print("No data available for the specified ticker and date range.")

# %%
