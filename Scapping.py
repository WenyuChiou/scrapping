# Scraping historical data for certain stocks
# The recommended modulus for scraping historical stock data include yfinance, ffn, and FinMind

#%%%
import yfinance as yf # suitablie for US stock market
from FinMind.data import DataLoader  # worked for futrue prices
import ffn
import pandas as pd
import os
from talib import abstract
import talib
from Indicator import indicator

#%% Scraping future prices via FinMind
from FinMind.data import DataLoader
import pandas as pd

# Initialize DataLoader
dl = DataLoader()

# Define the parameters for scraping
futures_id = 'TX'  # TX is the symbol for Taiwan futures
start_date = '1998-01-01'

try:
    # Scraping the futures data
    print(f"Scraping future data for {futures_id} starting from {start_date}...")
    future_data = dl.taiwan_futures_daily(futures_id=futures_id, start_date=start_date)

    if future_data.empty:
        print(f"No data found for futures ID {futures_id}. Please verify the ID or date.")
    else:
        # Removing after-market data
        future_data = future_data[future_data.trading_session == "position"]  # Keep only position data
        
        # Removing rows without settlement data
        future_data = future_data[future_data.settlement_price > 0]
        
        # Keeping only the near-month contract data
        future_data = future_data[future_data['contract_date'] == future_data.groupby('date')['contract_date'].transform('min')]
        
        # Print success message
        print(f"Successfully scraped future data for {futures_id}. Number of records: {len(future_data)}")

except Exception as e:
    print(f"Error occurred while scraping futures data: {e}")

# Optionally, save the data to a CSV file for further analysis
# future_data.to_csv("taiwan_futures_data.csv", index=False)


#%% Scraping future prices via FinMind using a function
from FinMind.data import DataLoader
import pandas as pd

# Define the function to scrape Taiwan futures data
def scrape_taiwan_futures(futures_id, start_date):
    """
    Function to scrape Taiwan futures data using FinMind API.
    
    Parameters:
        futures_id (str): The ID of the futures to scrape data for (e.g., 'TX').
        start_date (str): The start date for scraping in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: A DataFrame containing the scraped futures data.
    """
    dl = DataLoader()
    
    try:
        # Scraping the futures data
        print(f"Scraping future data for {futures_id} starting from {start_date}...")
        future_data = dl.taiwan_futures_daily(futures_id=futures_id, start_date=start_date)

        if future_data.empty:
            print(f"No data found for futures ID {futures_id}. Please verify the ID or date.")
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

        # Removing after-market data
        future_data = future_data[future_data.trading_session == "position"]  # Keep only position data
        
        # Removing rows without settlement data
        future_data = future_data[future_data.settlement_price > 0]
        
        # Keeping only the near-month contract data
        future_data = future_data[future_data['contract_date'] == future_data.groupby('date')['contract_date'].transform('min')]
        
        # Print success message
        print(f"Successfully scraped future data for {futures_id}. Number of records: {len(future_data)}")

        return future_data

    except Exception as e:
        print(f"Error occurred while scraping futures data for {futures_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs

# Example usage of the function
if __name__ == "__main__":
    futures_id = 'TX'  # TX is the symbol for Taiwan futures
    start_date = '1998-01-01'

    # Scrape the futures data
    future_data = scrape_taiwan_futures(futures_id, start_date)

    # Optionally, save the data to a CSV file for further analysis
    if not future_data.empty:
        future_data.to_csv("taiwan_futures_data.csv", index=False)



#%% Import required libraries
import ffn
import talib
import os
import pandas as pd
from talib import abstract

# Directory for saving files
dir = r"C:\Users\user\Desktop\investment\python\scrapping\scraping_data"
os.makedirs(dir, exist_ok=True)

#%% Function to scrape prices and add indicators
def scrape_and_analyze_prices(ticker, start_date, end_date):
    prices = ffn.get([ticker], start=start_date, end=end_date)
    if prices.empty:
        print(f"No data found for {ticker}.")
        return pd.DataFrame()
    
    # Add SMA and Bollinger Bands
    for period in [20, 50, 100, 150]:
        prices[f'{ticker}_sma_{period}'] = talib.SMA(prices[ticker].values, timeperiod=period)
    upper, middle, lower = talib.BBANDS(prices[ticker].values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    prices[['upperband_5day', 'middleband_5day', 'lowerband_5day']] = upper, middle, lower

    # Add MACD
    macd, signal, hist = talib.MACD(prices[ticker].values, fastperiod=12, slowperiod=26, signalperiod=9)
    prices[['macd', 'macdsignal', 'macdhist']] = macd, signal, hist

    csv_filename = os.path.join(dir, f'{ticker}.csv')
    prices.to_csv(csv_filename)
    print(f"Data saved to {csv_filename}")

    return prices

#%% Indicator Class
class Indicator:
    def __init__(self, stock_df):
        self.df = stock_df

    def add_indicator(self, indicator_list, setting=None):
        for ind in indicator_list:
            output = eval(f'abstract.{ind}(self.df)' if setting is None else f'abstract.{ind}(self.df, {setting.get(ind)})')
            output.name = ind.lower() if isinstance(output, pd.Series) else None
            self.df = pd.merge(self.df, pd.DataFrame(output), left_on=self.df.index, right_on=output.index)
        return self.df.set_index(self.df.columns[0])

#%% Main script
if __name__ == "__main__":
    prices = scrape_and_analyze_prices('spx', '2021-01-01', '2024-01-01')

    #custom_settings = {
    #'RSI': "{'timeperiod': 14}",
    #'MACD': "{'fastperiod': 10, 'slowperiod': 30, 'signalperiod': 8}"}
    if not prices.empty:
        cc = Indicator(prices)
        updated_df = cc.add_indicator(['RSI', 'MACD'])
        updated_df.to_csv(os.path.join(dir, 'spx_with_indicators.csv'))
        print(f"Updated data saved to {os.path.join(dir, 'spx_with_indicators.csv')}")


# %%
