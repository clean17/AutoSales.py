import os
import pandas as pd
from datetime import datetime, timedelta
from utils import fetch_stock_data, get_kor_ticker_dict_list
import time

start = time.time()   # 시작 시간(초)
nowTime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
print(f'{nowTime}: running 10_update_stock_data.py...')


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
os.makedirs(pickle_dir, exist_ok=True) # 없으면 생성

DATA_COLLECTION_PERIOD = 700 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


# chickchick.com에서 종목 리스트 조회
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())

for count, ticker in enumerate(tickers):
    time.sleep(3)
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    # if count % 100 == 0:
    #     print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    # 데이터 다운로드
    data = fetch_stock_data(ticker, start_date, today)

    if data is None or len(data) == 0:
        print(f"⚠️ 데이터 없음: {ticker} ({stock_name})")
        continue

    # 파일 저장 (기본적으로 같은 경로면 덮어쓰기)
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    data.to_pickle(filepath)
    df = pd.read_pickle(filepath)
    # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}] - {len(df)}")

end = time.time()     # 끝 시간(초)
elapsed = end - start
print(f"10_update_stock_data.py 총 소요 시간: {elapsed:.2f}초")