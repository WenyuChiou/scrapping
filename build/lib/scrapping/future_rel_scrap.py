# %%
import requests
import pandas as pd
from FinMind.data import DataLoader
import os

class TaiwanFuturesData:
    def __init__(self, input_df=None):
        self.api = DataLoader()
        self.df = pd.DataFrame()
        self.open_interest_df = pd.DataFrame()
        self.input_df = input_df

    def fetch_data(self, data_id, start_date, end_date):
        """
        抓取期貨三大法人籌碼(成交口數、金額等等)
        資料時間從2018-06-05開始
        
        From FinMind
        
        Parameter:
            data_id: future ID (str)
            start_date: the start of date (datatime); format: 'yyyy-mm-dd'
            end_data: the end of date (datetime.date); format: 'yyyy-mm-dd'
        
        Return:
            pivot_df: table (Dataframe)
        """
        try:
            self.df = self.api.taiwan_futures_institutional_investors(
                data_id=data_id,
                start_date='2018-06-05',
                end_date=end_date,
            )
            if not self.df.empty:
                self.df = self.df.rename(columns={
                    'futures_id': '期貨代號',
                    'date': '日期',
                    'institutional_investors': '投資者類型',
                    'long_deal_volume': '多單成交口數',
                    'long_deal_amount': '多單成交金額',
                    'short_deal_volume': '空單成交口數',
                    'short_deal_amount': '空單成交金額',
                    'long_open_interest_balance_volume': '多單未平倉口數',
                    'long_open_interest_balance_amount': '多單未平倉金額',
                    'short_open_interest_balance_volume': '空單未平倉口數',
                    'short_open_interest_balance_amount': '空單未平倉金額'
                })
                # 將數據按投資者類型分成不同列
                pivot_df = self.df.pivot_table(
                    index=['期貨代號', '日期'],
                    columns='投資者類型',
                    values=['多單成交口數', '多單成交金額', '空單成交口數', '空單成交金額', '多單未平倉口數', '多單未平倉金額', '空單未平倉口數', '空單未平倉金額'],
                    aggfunc='first'
                )
                pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
                pivot_df = pivot_df.reset_index().set_index('日期')
                return pivot_df
            else:
                print("未找到任何數據")
        except Exception as e:
            print(f"抓取數據時發生錯誤: {e}")

    def fetch_open_interest_data_big(self, data_id, start_date, end_date, token=None):
        """
        抓取前五大十大交易人籌碼(成交口數、金額等等)
        資料時間從2018-06-05開始
        From FinMind
        
        Parameter:
            data_id: future ID (str)
            start_date: the start of date (datatime); format: 'yyyy-mm-dd'
            end_data: the end of date (datetime.date); format: 'yyyy-mm-dd'
            token: VPN for FinMind
        Return:
            self.open_interest_df_big: table (dataframe)
        
        """
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            parameter = {
                "dataset": "TaiwanFuturesOpenInterestLargeTraders",
                "data_id": data_id,
                "start_date": start_date,
                "end_date": end_date,
                "token": token,
            }
            response = requests.get(url, params=parameter)
            data = response.json()
            if "data" in data and data["data"]:
                self.open_interest_df_big = pd.DataFrame(data["data"]).query('contract_type == "all"').set_index('date')
                self.open_interest_df_big = self.open_interest_df_big.rename(columns={
                    'date': '日期',
                    'market_open_interest': '全體未平倉量',
                    'contract_type': '合約類型',
                    'buy_top5_trader_open_interest': '前五大交易員買方未平倉量',
                    'buy_top5_trader_open_interest_per': '前五大交易員買方未平倉量百分比',
                    'buy_top10_trader_open_interest': '前十大交易員買方未平倉量',
                    'buy_top10_trader_open_interest_per': '前十大交易員買方未平倉量百分比',
                    'sell_top5_trader_open_interest': '前五大交易員賣方未平倉量',
                    'sell_top5_trader_open_interest_per': '前五大交易員賣方未平倉量百分比',
                    'sell_top10_trader_open_interest': '前十大交易員賣方未平倉量',
                    'sell_top10_trader_open_interest_per': '前十大交易員賣方未平倉量百分比',
                    'name': '商品名稱',
                    'futures_id': '期貨代號'
                })
                return self.open_interest_df_big
            else:
                print("未找到全體未平倉量數據")
        except Exception as e:
            print(f"抓取全體未平倉量數據時發生錯誤: {e}")
        
        return pd.DataFrame()

    def fetch_open_interest_data(self, futures_id, start_date, end_date):
        """
        抓取全體未平倉數據
        
        From FinMind
        
        Parameter:
            data_id: future ID (str)
            start_date: the start of date (datatime); format: 'yyyy-mm-dd'
            end_data: the end of date (datetime.date); format: 'yyyy-mm-dd'
            token: VPN for FinMind
        Return:
            self.open_interest_df: table (dataframe)
        
        """        
        try:
            self.open_interest_df = self.api.taiwan_futures_daily(
                futures_id=futures_id,
                start_date=start_date,
                end_date=end_date,
            )
            if not self.open_interest_df.empty:
                self.open_interest_df = self.open_interest_df.rename(columns={
                    'date': '日期',
                    'futures_id': '契約',
                    'delivery_month': '到期月份(週別)',
                    'open': '開盤價',
                    'high': '最高價',
                    'low': '最低價',
                    'close': '收盤價',
                    'change': '漲跌價',
                    'change_percent': '漲跌%',
                    'volume': '成交量',
                    'settlement_price': '結算價',
                    'open_interest': '未沖銷契約數',
                    'trading_session': '交易時段'
                })
                return self.open_interest_df
            else:
                print("未找到全體未平倉量數據")
        except Exception as e:
            print(f"抓取全體未平倉量數據時發生錯誤: {e}")
        
        return pd.DataFrame()

    def calculate_long_short_ratio(self, data_id, start_date, end_date):
        self.fetch_open_interest_data(data_id, start_date, end_date)
        self.fetch_data(data_id, start_date, end_date)

        if self.open_interest_df.empty:
            print("未找到全體未平倉量數據，無法繼續計算。")
            return None

        total_open_interest_grouped = self.open_interest_df.groupby('日期').agg({'未沖銷契約數': 'sum'})
        institutional_df = self.df[self.df['投資者類型'].isin(['外資', '自營商', '投信'])]
        if institutional_df.empty:
            print("沒有找到三大法人的數據。")
            return None

        institutional_grouped = institutional_df.groupby('日期').agg({'多單未平倉口數': 'sum', '空單未平倉口數': 'sum'})

        merged_df = pd.merge(institutional_grouped, total_open_interest_grouped, on='日期')
        merged_df['小台三大法人未平倉量'] = merged_df['多單未平倉口數'] + merged_df['空單未平倉口數']
        merged_df['散戶多單'] = merged_df['未沖銷契約數'] - merged_df['多單未平倉口數']
        merged_df['散戶空單'] = merged_df['未沖銷契約數'] - merged_df['空單未平倉口數']
        merged_df['多空比'] = (merged_df['散戶多單'] - merged_df['散戶空單']) / merged_df['未沖銷契約數']

        print("\n小台期貨多空比計算結果：")
        print(merged_df[[ '小台三大法人未平倉量', '未沖銷契約數', '散戶多單', '散戶空單', '多空比']])
        return merged_df
    
    def merge_data(self, data_ID, start_date, end_date = None):
        if end_date is None:
            end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        
        df_1 = self.fetch_data(data_id=data_ID, start_date=start_date, end_date=end_date)
        
        df_2 = self.fetch_open_interest_data_big(data_id=data_ID, start_date=start_date, end_date=end_date)

        df_3 = self.fetch_open_interest_data(futures_id=data_ID, start_date=start_date, end_date=end_date)
        if not df_3.empty:
            df_3 = df_3.groupby('日期').agg({'未沖銷契約數': 'sum'})
                    
        df_4 = self.calculate_long_short_ratio(data_id='MTX', start_date=start_date, end_date=end_date)
        

        # 合併數據，以日期作為索引，去掉合約類型和商品名稱欄位，並去掉有 NaN 的行
        if df_1 is not None and not df_1.empty and df_2 is not None and not df_2.empty and df_3 is not None and not df_3.empty and df_4 is not None:
            print('Processing data')

            combined_df = pd.concat([self.input_df,df_1, df_2, df_3, df_4], axis=1, join='outer').drop(columns=['合約類型', '商品名稱','期貨代號']).dropna(subset="open")

        else:
            print("Lack of data")
        return combined_df
           
#%%
if __name__ == "__main__":
    futures_data = TaiwanFuturesData()
    save_path = r'C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\scrapping\scraping data\taiwan_futures_data.csv'
    start_date = '2018-06-05'
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    # df_1 = futures_data.fetch_data(data_id='TX', start_date=start_date, end_date=end_date)
    # if df_1 is not None:
    #     df_1.to_csv(os.path.join(os.path.dirname(save_path), f"三大法人_20180601_TX.csv"), encoding='utf-8-sig')
    
    # df_2 = futures_data.fetch_open_interest_data_big(data_id='TX', start_date=start_date, end_date=end_date)
    # if not df_2.empty:
    #     df_2.to_csv(os.path.join(os.path.dirname(save_path), f"十大交易人_{start_date}_TX.csv"), encoding='utf-8-sig')
    
    # df_3 = futures_data.fetch_open_interest_data(futures_id='TX', start_date=start_date, end_date=end_date)
    # if not df_3.empty:
    #     df_3 = df_3.groupby('日期').agg({'未沖銷契約數': 'sum'})
    #     df_3.to_csv(os.path.join(os.path.dirname(save_path), f"未沖銷契約數_{start_date}_TX.csv"), encoding='utf-8-sig')
    
    # df_4 = futures_data.calculate_long_short_ratio(data_id='MTX', start_date=start_date, end_date=end_date, futures_id='MTX')
    # if df_4 is not None:
    #     df_4.to_csv(os.path.join(os.path.dirname(save_path), f"小台多空比_20180601_TX.csv"), encoding='utf-8-sig')

    # combined_df = pd.concat([df_1, df_2, df_3, df_4], axis=1, join='outer')
    # # 合併數據，以日期作為索引，去掉合約類型和商品名稱欄位，並去掉有 NaN 的行
    # if df_1 is not None and not df_1.empty and df_2 is not None and not df_2.empty and df_3 is not None and not df_3.empty and df_4 is not None:
    #     combined_df = pd.concat([df_1, df_2, df_3, df_4], axis=1, join='outer').drop(columns=['合約類型', '商品名稱','期貨代號']).dropna()
    #     combined_df.to_csv(os.path.join(os.path.dirname(save_path), f"合併_20180601_TX.csv"), encoding='utf-8-sig')

# %%
