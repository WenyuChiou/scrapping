#%%
# main_trading_script.py

import os
from scraping_and_indicators import StockDataScraper
from plot_with_sma_and_volume import plot_with_sma_and_volume
from scrape_taiwan_futures import Future_Taiwan_prices
import pandas as pd
from future_rel_scrap import TaiwanFuturesData
from TAIndicator import TAIndicatorSettings
from alpha_eric import AlphaFactory

#%%
# from sma_trading_strategy import sma_trading_strategy  # Importing the trading function from the separate file

# Directory for saving files
dir = r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\scraping data"
os.makedirs(dir, exist_ok=True)

# Step 1: Scrape historical data for a given stock ticker
ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2024-10-26'
scraper = StockDataScraper()
prices = scraper.scrape_prices_yfinance(ticker, start_date, end_date)

# Step 2: Add specified indicators if data is present
if not prices.empty:
    
    indicator_settings = TAIndicatorSettings()
    filtered_settings, timeperiod_only_indicators = indicator_settings.process_settings()  # 处理所有步骤并获取结果
    
    
    # Add SMA indicators with custom periods
    prices_with_sma = scraper.add_indicator_with_timeperiods(prices,timeperiod_only_indicators ,timeperiods=[5, 10, 20, 50, 100, 200])

    # Define the other indicators to add and custom settings
    indicator_list = list(filtered_settings.keys())
    # indicator_list = ['RSI', 'BBANDS', 'MACD', 'ATR']
    # custom_settings = {
    #     'RSI': {'timeperiod': 14},   # Custom RSI period
    #     'BBANDS': {'timeperiod': 20, 'nbdevup': 2.0, 'nbdevdn': 2.0, 'price': 'close'},  # Custom Bollinger Bands settings
    #     'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},  # Custom MACD settings
    #     'ATR': {'timeperiod': 14},  # Custom ATR settings
    #     'KD' : {'fastk_period': 9}  # KD settings
    # }

    # Add other indicators using the add_specified_indicators function
    prices_with_indicators = scraper.add_specified_indicators(prices_with_sma, indicator_list, filtered_settings)

    prices_with_indicators = scraper.calculate_profit_or_loss(prices_with_indicators, 20, column_name= 'adj_close')  
 
    future_data_with_indicators = scraper.calculate_avg_return(prices_with_indicators,20, column_name= 'close')      
    # Step 3: Save the updated DataFrame with indicators to a CSV file
    csv_filename = os.path.join(dir, f'{ticker}__{start_date}_with_specified_indicators_yfinance.csv')
    prices_with_indicators.to_csv(csv_filename)
    print(f"Updated data saved to {csv_filename}")

    # Step 4: Plot price with SMA and volume
    plot_with_sma_and_volume(prices_with_indicators, ticker, [5, 10, 20, 50, 100, 200])
    
else:
    print("No data available for the specified ticker and date range.")

# %%
import os
from scraping_and_indicators import StockDataScraper
from plot_with_sma_and_volume import plot_with_sma_and_volume
from scrape_taiwan_futures import Future_Taiwan_prices
import pandas as pd
from future_rel_scrap import TaiwanFuturesData
from TAIndicator import TAIndicatorSettings
from alpha_eric import AlphaFactory

# Example usage of the function
if __name__ == "__main__":
    futures_id = 'TX'  # TX is the symbol for Taiwan futures
    start_date = '2018-10-05'
    end_date = '2024-11-20'
    save_path = r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\scraping data\taiwan_futures_data.csv'

    indicator_settings = TAIndicatorSettings()
    filtered_settings, timeperiod_only_indicators = indicator_settings.process_settings()  # 处理所有步骤并获取结果
    
    # Scrape the futures data
    future_data = Future_Taiwan_prices(futures_id=futures_id, start_date=start_date, end_data=end_date)
    scarping_data = future_data.scrape_taiwan_futures()
    scraper = StockDataScraper()



    if not scarping_data.empty:
        # Add SMA indicators with custom periods
        
        prices_with_sma = scraper.add_indicator_with_timeperiods(scarping_data,timeperiod_only_indicators, timeperiods=[5, 10, 20, 50, 100, 200])

    # Define the other indicators to add and custom settings
        indicator_list = list(filtered_settings.keys())

        # Add other indicators using the add_specified_indicators function
        future_data_with_indicators = scraper.add_specified_indicators(prices_with_sma, indicator_list, filtered_settings)

        future_data_with_indicators = scraper.calculate_profit_or_loss(future_data_with_indicators,20, column_name= 'close')

        future_data_with_indicators = scraper.calculate_avg_return(future_data_with_indicators,20, column_name= 'close')
        
        future_data_with_indicators = scraper.process_taiwan_margin_purchase_short_sale(start_date=start_date,end_date=end_date,additional_df=future_data_with_indicators)
        
        alpha = AlphaFactory(future_data_with_indicators)
        
        future_data_with_indicators = alpha.add_all_alphas(days= [5,10,20,60,120,240],
                                                      custom_params={"alpha01":{'par':{'alpha':1/2,'beta':2,'theta':1.5}}})
        
        
        #add alpha01 of different form
        future_data_with_indicators = alpha.alpha01(days= [5,10,20,60,120,240],
                                                     par={'alpha':1/2,'beta':2,'theta':1.5},type=True)
        
    
        
        # Save the updated DataFrame with indicators to a CSV file
        updated_save_path = os.path.join(os.path.dirname(save_path), f"taiwan_futures_data_with_indicators_{start_date}.csv")
        
        future_data_with_indicators.to_csv(updated_save_path)
        print(f"Updated data saved to {updated_save_path}")
        
            
        # Step 4: Plot price with SMA and volume
        plot_with_sma_and_volume(future_data_with_indicators, futures_id, [5, 10, 20, 50, 100, 200])
        # Drop rows with NaN values
        
        # data_cleaned = pd.concat([future_data_with_indicators.iloc[:-100].dropna(), future_data_with_indicators.iloc[-100:]])


        # data_cleaned = scraper.filter_data_by_date_range(data_cleaned, '2018-10-05','2024-11-12')
        future_data_with_indicators.index = future_data_with_indicators.index.strftime('%Y-%m-%d')
        related_data = TaiwanFuturesData(input_df=future_data_with_indicators)
        gg = related_data.merge_data('TX',start_date=start_date, end_date=end_date)
        gg.to_excel(os.path.join(os.path.dirname(save_path), "taiwan_futures_data_with_indicators_all.xlsx"))

# %%
