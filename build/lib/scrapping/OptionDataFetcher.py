#%%
from FinMind.data import DataLoader
import pandas as pd
import requests


class OptionDataFetcher:
    def __init__(self, api_token=None):
        self.api = DataLoader()
        self.api_token = api_token
        self.df = None

    def fetch_option_institutional_investors_data(self, data_id, start_date, end_date=None):
        """
        抓取期權三大法人籌碼(成交口數、金額等等)
        資料時間從2018-06-05開始

        參數:
            data_id: 期權代號 (str)
            start_date: 開始日期 (str); 格式: 'yyyy-mm-dd'
            end_date: 結束日期 (str); 格式: 'yyyy-mm-dd'

        回傳:
            pivot_df: 數據表 (Dataframe)
        """
            
        try:
            
            if not end_date:
                end_date = end_date    
            else:
                end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
            
            df = self.api.taiwan_option_institutional_investors(
                data_id=data_id,
                start_date=start_date,
                end_date=end_date,
            )
            if df.empty:
                print("未找到任何數據")
                return None

            df = df.rename(columns={
                'option_id': '期權代號',
                'date': '日期',
                'call_put': '買賣權',
                'institutional_investors': '投資者類型',
                'contract_type': '合約類型',
                'long_open_interest_balance_volume': '多單未平倉口數',
                'short_open_interest_balance_volume': '空單未平倉口數'
            })

            # 將數據按投資者類型和買賣權分成不同列
            df = df[df['投資者類型'] != '投信']
            pivot_df = df.pivot_table(
                index='日期',
                columns=['投資者類型', '買賣權'],
                values=['多單未平倉口數', '空單未平倉口數'],
                aggfunc='first'
            )
            pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

            # 計算 PC_ratio_三大 (自營商 + 外資)
            pivot_df['PC_ratio_三大'] = pivot_df[['空單未平倉口數_自營商_賣權', '空單未平倉口數_外資_賣權']].sum(axis=1) / pivot_df[['多單未平倉口數_自營商_買權', '多單未平倉口數_外資_買權']].sum(axis=1)

            return pivot_df
        except Exception as e:
            print(f"抓取數據時發生錯誤: {e}")
            return None

    def fetch_open_interest_data_all(self, data_id, start_date, end_date=None):
        """
        抓取期權未平倉數據，只要合約類型為 'all' 的資料

        Parameter:
            data_id: option ID (str)
            start_date: the start of date (str); format: 'yyyy-mm-dd'
            end_date: the end of date (str); format: 'yyyy-mm-dd'

        Return:
            filtered_df: table (Dataframe)
        """

        if not end_date:
            end_date = end_date    
        else:          
            end_date = pd.to_datetime('today').strftime('%Y-%m-%d')     
               
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanOptionOpenInterestLargeTraders",
            "data_id": data_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": self.api_token,
        }
        try:
            resp = requests.get(url, params=parameter)
            data = resp.json()
            df = pd.DataFrame(data['data'])

            # 過濾掉合約類型不是 'all' 的資料
            filtered_df = df[df['contract_type'] == 'all']

            # 將數據按買賣權分成不同列
            pivot_df = filtered_df.pivot_table(
                index='date',
                columns='put_call',
                values=[
                    'buy_top5_trader_open_interest',
                    'buy_top5_trader_open_interest_per',
                    'buy_top10_trader_open_interest',
                    'buy_top10_trader_open_interest_per',
                    'sell_top5_trader_open_interest',
                    'sell_top5_trader_open_interest_per',
                    'sell_top10_trader_open_interest',
                    'sell_top10_trader_open_interest_per',
                    'market_open_interest',
                    'buy_top5_specific_open_interest',
                    'buy_top5_specific_open_interest_per',
                    'buy_top10_specific_open_interest',
                    'buy_top10_specific_open_interest_per',
                    'sell_top5_specific_open_interest',
                    'sell_top5_specific_open_interest_per',
                    'sell_top10_specific_open_interest',
                    'sell_top10_specific_open_interest_per'
                ],
                aggfunc='first'
            )
            pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]


            # 計算 PC Ratio
            if 'market_open_interest_put' in pivot_df.columns and 'market_open_interest_call' in pivot_df.columns:
                pivot_df['PC_ratio'] = pivot_df['market_open_interest_put'] / pivot_df['market_open_interest_call']

            return pivot_df
        except Exception as e:
            print(f"抓取未平倉數據時發生錯誤: {e}")
            return None
        
    def merge_data(self, data_id, start_date, end_date=None):
        """
        合併 fetch_data 和 fetch_open_interest_data_all 的結果

        參數:
            data_id: 期權代號 (str)
            start_date: 開始日期 (str); 格式: 'yyyy-mm-dd'
            end_date: 結束日期 (str); 格式: 'yyyy-mm-dd'

        回傳:
            merged_df: 合併後的數據表 (Dataframe)
        """
        if not end_date:
            end_date = end_date
        else:
            end_date = pd.to_datetime('today').strftime('%Y-%m-%d') 
            
        df1 = self.fetch_option_institutional_investors_data(data_id, start_date, end_date)
        df2 = self.fetch_open_interest_data_all(data_id, start_date, end_date)

        if df1 is not None and df2 is not None:
            merged_df = pd.concat([df1, df2], axis=1, join='outer')
            return merged_df.dropna(how='all')
        else:
            print("合併數據時缺少資料")
            return None
# 使用範例
fetcher = OptionDataFetcher(api_token=None)
# pivot_data = fetcher.fetch_option_institutional_investors_data(data_id="TXO", start_date="2020-04-01", end_date="2022-04-02")
# if pivot_data is not None:
#     print(pivot_data.head())
# open_interest_data = fetcher.fetch_open_interest_data_all(data_id="TXO", start_date="2024-09-01")
# if open_interest_data is not None:
#     print(open_interest_data.head())
merged_data = fetcher.merge_data(data_id="TXO", start_date="2020-04-01")

# %%
