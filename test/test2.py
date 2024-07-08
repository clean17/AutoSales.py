import pandas as pd
from pykrx import stock

pd.set_option('display.max_columns', None)  # 모든 column 출력
#pd.set_option('display.max_rows', None)    # 모든 row 출력
 
# df = stock.get_market_fundamental("20210108")
# print(df.head(2))

tickers = stock.get_market_ticker_list("20190225", market="KOSDAQ")
print(tickers)

for ticker in stock.get_market_ticker_list():
        종목 = stock.get_market_ticker_name(ticker)
        print(종목)