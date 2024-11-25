#%%
# 匯入模組
from ib_insync import *
util.startLoop() # 開啟 Socket 線程
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=789)


#%%
import pandas as pd
# 先取得帳戶總覽，'DU228378'是我的 demo 帳號代碼，記得要改成你的 demo 帳號代碼，在 TWS 的右上方尋找
account_summary = ib.accountSummary(account='DU9954083')
# 再透過 pandas 轉換為 DataFrame
account_summary_df = pd.DataFrame(account_summary).set_index('tag')
#取得 Cash 現金的數字
account_summary_df.loc['AvailableFunds']
#取得 Securities Gross Position Value 持有中資產的帳面價值
account_summary_df.loc['GrossPositionValue']
#取得 Net Liquidation Value 帳戶清算的總價值
account_summary_df.loc['NetLiquidation']

#%%# 定義 contract 合約
contract = Contract(
    secType='STK',      # 買進的是「股票」，就鍵入「STK」
    symbol='TSLA',      # 鍵入股票代碼
    exchange='SMART',   # SMART 指的是 TWS 的智能路由，
                        # 它會根據以下條件找出最佳的交易所
                        # 1.交易所成本最小化
                        # 2.執行機率最大化
    currency='USD'      # 鍵入貨幣單位
)
# 定義 order 委託單
order = Order(
    action='BUY',       # 買進的話就是「BUY」，賣出/放空則是「SELL」
    totalQuantity=50,   # 這裡要鍵入的是「股」數喔！
    orderType='MKT'     # 例1下的是「市價單」，參數就是「MKT」
)
# 向 TWS 發送委託單！
ib.placeOrder(contract, order)

#%%
#例2:「用限價 360 買進 100 股 QQQ」
# 定義 contract 合約
contract = Contract(
    secType='STK',      # 買進「ETF」的話也是鍵入「STK」
    symbol='QQQ',       # 鍵入 ETF 代碼
    exchange='SMART',
    currency='USD'
)
# 定義 order 委託單
order = Order(
    action='BUY',
    totalQuantity=100,  # 這裡要鍵入的是「股」數喔！
    orderType='LMT',    # 例2下的是「限價單」，參數就是「LMT」
    lmtPrice=370,       # ★ 限價單會多一個參數，設定「掛單價格」★
)
# 向 TWS 發送委託單！
ib.placeOrder(contract, order)
#%%
# 透過這個函數，可以確認交易執行的情況
ib.executions()
#%%
# 透過這個函數，可以取得未執行完畢的交易委託
open_trades = ib.openTrades()
#檢視了一下 open_trades 會發現資訊量過多，各位針對自己所需去取值即可。這裡簡單做個示範，整理出一個記載重點資訊的 DataFrame
# 寫函數，從 open_trades 中的每一筆物件取值
def open_trade_info(trade_object):
    return {
        'orderId': trade_object.order.orderId,
        'action': trade_object.order.action,
        'totalQuantity': trade_object.order.totalQuantity,
        'orderType': trade_object.order.orderType,
        'lmtPrice': trade_object.order.lmtPrice,
        'secType': trade_object.contract.secType,
        'symbol': trade_object.contract.symbol
    }
    
# 整理成 DataFrame
open_trades_df = pd.DataFrame(list(map(lambda x: open_trade_info(x), open_trades)))
print(open_trades_df)
#%%

# ♦ 例5:「修改」一筆委託單
# 假設要修改剩下來的那筆委託單:
# → QQQ 買 80 股就好，不買 100 股了
# → 限價股價也從 370 改成 375 好了
# 以上要怎麼修改呢？
# # 首先我們需要辨識出要修的這筆委託單的專屬代號
loc_index = open_trades_df.index[open_trades_df.symbol == 'QQQ'][0]
modify_id = open_trades_df.loc[loc_index, 'orderId']
# 跟前面一樣定義 contract 以及 order，但這次 order 裡要增加 orderId 這參數
contract = Contract(
    secType='STK',
    symbol='QQQ',
    exchange='SMART',
    currency='USD'
)
order = Order(
    orderId=modify_id,  # ★ 鍵入剛剛找到的專屬代號 ★
    action='BUY',
    totalQuantity=80,   # 從 100 股修改成 80 股
    orderType='LMT',
    lmtPrice=375,       # 從限價 370 修改成 375
)
# 向 TWS 發送，就能對正在掛著的委託單進行修改了！
ib.placeOrder(contract, order)
# %%
# ♦ 例6:「取消」一筆委託單
# 假設要把一筆委託單取消掉，該怎麼做呢？
# 記得要定期更新 open_trades_df（見例4）
# 一樣要先辨識出該筆委託單的專屬代號
cancel_id = open_trades_df['orderId'][0]
# 定義 order 委託單，這次只要指出要取消委託單的專屬代號即可
order = Order(
    orderId=cancel_id   # ★ 鍵入該筆委託單的專屬代號 ★
)
# 向 TWS 發送 cancelOrder，取消掉正在掛著的委託單！
ib.cancelOrder(order)
# %%

# ♦ 例7: 更新取得投資組合的資訊
# # 透過這個函數，可以輕鬆取得投資組合的資訊
portfolio_data = ib.portfolio()
# 跟例4 非常相似，我們針對自己需求，將 portfolio_data 的資訊整理成 DataFrame
# 寫函數，從 portfolio_data 中的每一筆物件取值
def portfolio_info(asset_object):
    return {
        'symbol': asset_object.contract.symbol,
        'primaryExchange': asset_object.contract.primaryExchange,
        'currency': asset_object.contract.currency,
        'position': asset_object.position,
        'marketPrice': asset_object.marketPrice,
        'marketValue': asset_object.marketValue,
        'averageCost': asset_object.averageCost,
        'unrealizedPNL': asset_object.unrealizedPNL,
        'realizedPNL': asset_object.realizedPNL
    }
# 整理成 DataFrame
portfolio_data_df = pd.DataFrame(list(map(lambda x: portfolio_info(x), portfolio_data)))

#%%
import pandas as pd
from ib_insync import IB, Stock, Option, Contract, Future

# 初始化 IB 連線
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=65)



#%%
def collect_historical_futures_price(symbol, sec_type, currency, exchange, expiry, whatToShow='Bid_Ask'):
    try:
        # 創建期貨合約，並設定交易所、合約類型、貨幣和到期月份
        futures_contract = Future()
        futures_contract.symbol = symbol
        futures_contract.secType = sec_type
        futures_contract.currency = currency
        futures_contract.exchange = exchange
        futures_contract.lastTradeDateOrContractMonth = expiry
        qualified_futures = ib.qualifyContracts(futures_contract)[0]
        print(f"Future : {qualified_futures}")

        # 請求該股票的歷史市場數據
        historicalticks = ib.reqHistoricalTicks(
            qualified_futures,
            startDateTime='20241101-18:00:00',
            endDateTime='20241101-18:10:00',
            numberOfTicks=500,
            whatToShow='Bid_Ask',
            useRth=False
        )
   
        print("Data collected Successfully:", historicalticks)
        
        extracted_data_list = []
        
        # Extract Historical Tick Trades Data
        if whatToShow == 'Trades':
            for tick_data in historicalticks:
                extracted_data = {
                    'time': tick_data.time,
                    'pastLimit': tick_data.tickAttribLast.pastLimit,
                    'unreported': tick_data.tickAttribLast.unreported,
                    'price': tick_data.price,
                    'size': tick_data.size,
                    'exchange': tick_data.exchange,
                    'specialConditions': tick_data.specialConditions
                }
                extracted_data_list.append(extracted_data)
        
        # Extract Historical Tick Bid/Ask Data
        elif whatToShow == 'Bid_Ask':
            for tick_data in historicalticks:
                extracted_data = {
                    'time': tick_data.time,
                    'bidPastLow': tick_data.tickAttribBidAsk.bidPastLow,
                    'askPastHigh': tick_data.tickAttribBidAsk.askPastHigh,
                    'priceBid': tick_data.priceBid,
                    'priceAsk': tick_data.priceAsk,
                    'sizeBid': tick_data.sizeBid,
                    'sizeAsk': tick_data.sizeAsk
                }
                extracted_data_list.append(extracted_data)
        # Extract Historical Tick Midpoint data
        else:
            for tick_data in historicalticks:
                extracted_data = {
                    'time': tick_data.time,
                    'price': tick_data.price,
                    'size': tick_data.size
                }
                extracted_data_list.append(extracted_data)

        return pd.DataFrame(extracted_data_list)                
                  

    except Exception as e:
        print(f"獲取歷史期貨價格資料時發生錯誤: {e}")
        return pd.DataFrame()


# 示例：獲取期貨的歷史數據
historical_futures_df = collect_historical_futures_price(symbol='XINA50', sec_type='FUT', currency='USD', exchange='SGX', expiry='202412', whatToShow='Bid_Ask')
print(historical_futures_df)

# %%

