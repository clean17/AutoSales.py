import matplotlib
matplotlib.use('Agg')
import os
import sys
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from send2trash import send2trash
import ast
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data, add_technical_features, get_kor_ticker_list, check_column_types, get_safe_ticker_list
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import requests


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
LOOK_BACK = 15
os.makedirs(pickle_dir, exist_ok=True)

RATE_OF_CHANGE = 10
AVERAGE_TRADING_VALUE = 1_000_000_000 # 평균거래대금

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers = get_kor_ticker_list()
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}


# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
    #     data = fetch_stock_data(ticker, start_yesterday, today)
    #
    # # 중복 제거 & 새로운 날짜만 추가
    # if not df.empty:
    #     # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
    #     new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
    #     df = pd.concat([df, new_rows])
    # else:
    #     df = data

    # 너무 먼 과거 데이터 버리기
    if len(df) > 270:
        df = df.iloc[-270:]

    data = df[:-1]

    ########################################################################

    actual_prices = data['종가'].values # 종가 배열
    last_close = actual_prices[-1]

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 30:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 500원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row['종가'] < 500:
        # print("                                                        종가가 0이거나 500원 미만 → pass")
        continue

    # 최근 2주 거래대금이 기준치 이하면 패스
    recent_data = data.tail(10)
    recent_trading_value = recent_data['거래량'] * recent_data['종가']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        continue

    ########################################################################
    # ======== 조건 체크 시작 ========

    closes = data['종가'].values

    # 오늘/어제/10일전 인덱스 계산
    today_close = closes[-1]
    yesterday_close = closes[-2]
    close_10ago = closes[-12]

    # 1. 오늘 등락률 계산 (오늘 vs 어제)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    if change_pct_today < 10:
        continue  # 10% 미만은 제외

    # 2. 최근 10영업일간(오늘제외) 종가 변화율 (10일 전 vs 어제)
    close_past10 = closes[-12]  # 10일 전
    close_yesterday = closes[-2]  # 어제
    change_10days = (close_yesterday - close_past10) / close_past10 * 100

    if abs(change_10days) >= 4:
        continue  # 3% 이상 변화면 횡보 X

    # 부합하면 결과에 저장 (상승률, 종목명, 코드)
    results.append((change_pct_today, stock_name, ticker))





#######################################################################

# 내림차순 정렬 (상승률 기준)
results.sort(reverse=True, key=lambda x: x[0])

# 최대 종목명 길이 계산
max_name_len = max(len(name) for _, name, _ in results)

for change, stock_name, ticker in results:
    # 종목명을 왼쪽 정렬 + 공백 padding
    print(f"==== {stock_name.ljust(max_name_len)} [{ticker}] 상승률 {change:.2f}% ====")
