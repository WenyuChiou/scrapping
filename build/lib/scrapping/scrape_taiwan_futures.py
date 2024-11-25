#%% Scraping future prices via FinMind using a function
from FinMind.data import DataLoader
import pandas as pd

# Define the function to scrape Taiwan futures data
class Future_Taiwan_prices():
    def __init__(self, futures_id=None, start_date=None, end_data=None, save_path=None):
        self.futures_id = futures_id
        self.start_date = start_date
        self.end_date = end_data
        self.save_path = save_path
    
    def scrape_taiwan_futures(self):
        """
        Function to scrape Taiwan futures data using FinMind API.
        
        Parameters:
            save_path (str, optional): The path where the CSV file will be saved.
        
        Returns:
            pd.DataFrame: A DataFrame containing the scraped futures data.
        """
        dl = DataLoader()
        
        try:
            # Scraping the futures data
            print(f"Scraping future data for {self.futures_id} starting from {self.start_date}...")
            future_data = dl.taiwan_futures_daily(futures_id=self.futures_id, start_date=self.start_date, end_date=self.end_date)

            if future_data.empty:
                print(f"No data found for futures ID {self.futures_id}. Please verify the ID or date.")
                return pd.DataFrame()  # Return an empty DataFrame if no data is found
            
            # Removing rows without settlement data
            future_data = future_data[future_data.settlement_price > 0]
            
            # Keeping only the near-month contract data
            future_data = future_data[future_data['contract_date'] == future_data.groupby('date')['contract_date'].transform('min')]
            
            # Drop unnecessary columns 'open_interest' and 'trading_session'
            future_data.drop(columns=['open_interest', 'trading_session','futures_id','contract_date'], inplace=True)
            
            # Rename columns from 'max', 'min' to 'high', 'low'
            future_data.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
        
            future_data['date'] = pd.to_datetime(future_data['date']).dt.strftime('%Y-%m-%d')
        
            future_data.set_index('date', inplace=True)
            future_data.index = pd.to_datetime(future_data.index)
            
            # Print success message
            print(f"Successfully scraped future data for {self.futures_id}. Number of records: {len(future_data)}")

            # Save to CSV if save_path is provided
            if self.save_path:
                future_data.to_csv(self.save_path)
                print(f"Data saved to {self.save_path}")

            return future_data

        except Exception as e:
            print(f"Error occurred while scraping futures data for {self.futures_id}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if an error occurs

