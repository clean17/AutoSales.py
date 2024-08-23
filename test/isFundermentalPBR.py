from pykrx import stock
import pandas as pd
from datetime import datetime

# KOSPI와 KOSDAQ 시장의 종목 리스트 가져오기
tickers_kospi = stock.get_market_ticker_list(market="KOSPI")
tickers_kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
tickers = tickers_kospi + tickers_kosdaq

# PBR이 없는 종목 리스트
no_pbr_tickers = []

# 기준 날짜 설정 (예: 최근 데이터)
today = datetime.today().strftime('%Y%m%d')

# 각 종목의 펀더멘탈 데이터에서 PBR 확인
for ticker in tickers:
    daily_fundamental = stock.get_market_fundamental_by_date(today, today, ticker)

    # PBR 데이터가 없는지 확인
    if 'PBR' not in daily_fundamental.columns or pd.isna(daily_fundamental['PBR'].values[0]):
        stock_name = stock.get_market_ticker_name(ticker)
        no_pbr_tickers.append((ticker, stock_name))

# PBR이 없는 종목 출력
print("PBR이 없는 종목:")
for ticker, name in no_pbr_tickers:
    print(f"{ticker}: {name}")
