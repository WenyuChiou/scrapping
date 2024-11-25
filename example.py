#%%
# main_trading_script.py

import os
from scraping_and_indicators import scrape_prices_yfinance, add_specified_indicators, add_sma, calculate_profit_or_loss
from plot_with_sma_and_volume import plot_with_sma_and_volume
from scrape_taiwan_futures import scrape_taiwan_futures
import pandas as pd
#%%
# from sma_trading_strategy import sma_trading_strategy  # Importing the trading function from the separate file

# Directory for saving files
dir = r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\scraping data"
os.makedirs(dir, exist_ok=True)

# Step 1: Scrape historical data for a given stock ticker
ticker = 'NVDA'
start_date = '2023-01-01'
end_date = '2024-10-26'
prices = scrape_prices_yfinance(ticker, start_date, end_date)

# Step 2: Add specified indicators if data is present
if not prices.empty:
    # Add SMA indicators with custom periods
    prices_with_sma = add_sma(prices, timeperiods=[5, 10, 20, 50, 100, 200])

    # Define the other indicators to add and custom settings
    indicator_list = ['RSI', 'BBANDS', 'MACD', 'ATR']
    custom_settings = {
        'RSI': {'timeperiod': 14},   # Custom RSI period
        'BBANDS': {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'price': 'close'},  # Custom Bollinger Bands settings
        'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},  # Custom MACD settings
        'ATR': {'timeperiod': 14}  # Custom ATR settings
    }

    # Add other indicators using the add_specified_indicators function
    prices_with_indicators = add_specified_indicators(prices_with_sma, indicator_list, custom_settings)

    prices_with_indicators = calculate_profit_or_loss(prices_with_indicators, 10, column_name= 'adj_close')    
    # Step 3: Save the updated DataFrame with indicators to a CSV file
    csv_filename = os.path.join(dir, f'{ticker}__{start_date}_with_specified_indicators_yfinance.csv')
    prices_with_indicators.to_csv(csv_filename)
    print(f"Updated data saved to {csv_filename}")

    # Step 4: Plot price with SMA and volume
    plot_with_sma_and_volume(prices_with_indicators, ticker, [5, 10, 20, 50, 100, 200])
    
else:
    print("No data available for the specified ticker and date range.")

# %%
import pandas as pd
# Example usage of the function
if __name__ == "__main__":
    futures_id = 'TX'  # TX is the symbol for Taiwan futures
    start_date = '2024-01-01'
    save_path = r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\scraping data\taiwan_futures_data.csv'

    # Scrape the futures data
    future_data = scrape_taiwan_futures(futures_id, start_date, save_path)

    if not future_data.empty:
        # Add SMA indicators with custom periods
        
        prices_with_sma = add_sma(future_data, timeperiods=[5, 10, 20, 50, 100, 200])

        # Define the other indicators to add and custom settings
        indicator_list = ['RSI', 'BBANDS', 'MACD', 'ATR']
        custom_settings = {
            'RSI': {'timeperiod': 14},   # Custom RSI period
            'BBANDS': {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'price': 'close'},  # Custom Bollinger Bands settings
            'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},  # Custom MACD settings
            'ATR': {'timeperiod': 14}  # Custom ATR settings
            }
        # Add other indicators using the add_specified_indicators function
        future_data_with_indicators = add_specified_indicators(prices_with_sma, indicator_list, custom_settings)

        future_data_with_indicators = calculate_profit_or_loss(future_data_with_indicators,20, column_name= 'close')
        # Save the updated DataFrame with indicators to a CSV file
        updated_save_path = os.path.join(os.path.dirname(save_path), f"taiwan_futures_data_with_indicators_{start_date}.csv")
        future_data_with_indicators.to_csv(updated_save_path)
        print(f"Updated data saved to {updated_save_path}")
        
            
        # Step 4: Plot price with SMA and volume
        plot_with_sma_and_volume(future_data_with_indicators, futures_id, [5, 10, 20, 50, 100, 200])
# %%
