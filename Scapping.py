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
#%% Function to scrape prices and add indicators using yfinance
#%% Function to scrape prices and add indicators using yfinance
import yfinance as yf
import pandas as pd
import os
from talib import abstract

# Directory for saving files
dir = r"C:\Users\user\Desktop\investment\python\scrapping\scraping_data"
os.makedirs(dir, exist_ok=True)

def scrape_prices_yfinance(ticker, start_date, end_date):
    """
    Scrapes historical prices using yfinance and returns a DataFrame.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'YINN').
        start_date (str): Start date for scraping in the format 'YYYY-MM-DD'.
        end_date (str): End date for scraping in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame containing the stock price data.
    """
    try:
        # Step 1: Scrape historical price data
        print(f"Scraping historical data for {ticker} from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date)

        # Check if data is available
        if df.empty:
            print(f"No data found for {ticker}. Please verify the ticker symbol or date range.")
            return pd.DataFrame()

        # Step 2: Rename columns to lowercase to ensure consistency
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }, inplace=True)

        return df

    except Exception as e:
        print(f"Error occurred while scraping data for {ticker}: {e}")
        return pd.DataFrame()

#%%
def add_indicators(df, indicator_list, setting=None):
    """
    Adds specified technical indicators to the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock price data.
        indicator_list (list): List of indicators to add (e.g., ['SMA', 'RSI', 'BBANDS']).
        setting (dict, optional): Custom settings for indicators, e.g., {'SMA': {'timeperiod': 50}}.
        
    Returns:
        pd.DataFrame: Updated DataFrame with added indicators.
    """
    # Dictionary to specify available indicators in TA-Lib
    available_indicators = ['SMA', 'EMA', 'RSI', 'BBANDS', 'MACD']

    for ind in indicator_list:
        ind_upper = ind.upper()
        if ind_upper not in available_indicators:
            print(f"Warning: Indicator '{ind}' is not supported.")
            continue

        try:
            # Prepare the custom settings if provided, otherwise use default settings
            if setting is None or ind_upper not in setting:
                # Use the default settings
                output = eval(f"abstract.{ind_upper}(df)")
            else:
                # Use custom settings for the indicator
                output = eval(f"abstract.{ind_upper}(df, **setting[ind_upper])")

            # Handle different types of outputs from TA-Lib
            if isinstance(output, pd.Series):
                output.name = ind.lower()
                df = pd.concat([df, output], axis=1)
            elif isinstance(output, pd.DataFrame):
                output.columns = [f"{ind.lower()}_{col}" for col in output.columns]
                df = pd.concat([df, output], axis=1)
            elif isinstance(output, tuple) and ind_upper == 'MACD':
                # MACD returns a tuple (macd, signal, hist)
                macd, signal, hist = output
                df[f'{ind.lower()}_line'] = macd
                df[f'{ind.lower()}_signal'] = signal
                df[f'{ind.lower()}_hist'] = hist

        except Exception as e:
            print(f"Failed to add indicator '{ind}': {e}")

    return df



#%%
if __name__ == "__main__":
    # Step 1: Scrape historical data for a given stock ticker
    prices = scrape_prices_yfinance('YINN', '2021-01-01', '2024-01-01')

    # Step 2: Add indicators if data is present
    if not prices.empty:
        # Define the indicators to add and custom settings
        indicator_list = ['SMA', 'RSI', 'BBANDS']
        custom_settings = {
            'SMA': {'timeperiod': 50},
            'RSI': {'timeperiod': 14},
            'BBANDS': {'timeperiod': 5, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'matype': 0}
        }

        # Add indicators using the add_indicators function
        prices_with_indicators = add_indicators(prices, indicator_list, custom_settings)

        # Step 3: Save the updated DataFrame with indicators to a CSV file
        csv_filename = os.path.join(dir, 'yinn_with_indicators_yfinance.csv')
        prices_with_indicators.to_csv(csv_filename)
        print(f"Updated data saved to {csv_filename}")
    else:
        print("No data available for the specified ticker and date range.")

# %%
