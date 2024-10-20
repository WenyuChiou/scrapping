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
def add_specified_indicators(df, indicator_list, setting=None):
    """
    Adds specified technical indicators to the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock price data.
        indicator_list (list): List of indicators to add (e.g., ['SMA', 'RSI', 'BBANDS', 'MACD']).
        setting (dict, optional): Custom settings for indicators, e.g., {'SMA': {'timeperiod': 50}}.
        
    Returns:
        pd.DataFrame: Updated DataFrame with added indicators.
    """
    # Dictionary to specify available indicators in TA-Lib
    available_indicators = ['SMA', 'EMA', 'RSI', 'BBANDS', 'MACD', 'ADX', 'ATR', 'STOCH', 'WILLR', 'CCI', 'MOM', 'ROC', 'OBV', 'SAR']

    for ind in indicator_list:
        ind_upper = ind.upper()
        if ind_upper not in available_indicators:
            print(f"Warning: Indicator '{ind}' is not supported.")
            continue

        try:
            # Prepare to evaluate the indicator
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
            elif isinstance(output, tuple):
                # Handle multiple outputs for specific indicators like MACD, BBANDS, STOCH
                if ind_upper == 'MACD':
                    macd, signal, hist = output
                    df[f'{ind.lower()}_line'] = macd
                    df[f'{ind.lower()}_signal'] = signal
                    df[f'{ind.lower()}_hist'] = hist
                elif ind_upper == 'BBANDS':
                    upper, middle, lower = output
                    df[f'{ind.lower()}_upper'] = upper
                    df[f'{ind.lower()}_middle'] = middle
                    df[f'{ind.lower()}_lower'] = lower
                elif ind_upper == 'STOCH':
                    slowk, slowd = output
                    df[f'{ind.lower()}_slowk'] = slowk
                    df[f'{ind.lower()}_slowd'] = slowd
                else:
                    print(f"Warning: Multiple output indicator '{ind}' not handled correctly.")

        except Exception as e:
            print(f"Failed to add indicator '{ind}': {e}")

    return df

#%%
if __name__ == "__main__":
    # Step 1: Scrape historical data for a given stock ticker
    prices = scrape_prices_yfinance('YINN', '2021-01-01', '2024-01-01')

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
        csv_filename = os.path.join(dir, 'yinn_with_specified_indicators_yfinance.csv')
        prices_with_indicators.to_csv(csv_filename)
        print(f"Updated data saved to {csv_filename}")
    else:
        print("No data available for the specified ticker and date range.")
