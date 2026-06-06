"""
월-금 01시에 모든 종목의 지정한 기간동안의 OHLCV 데이터를 갱신
"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pandas as pd
from datetime import datetime, timedelta
from utils import fetch_stock_data, get_kor_ticker_dict_list, get_stock_name
import time

start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 10_update_stock_data.py...')


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")
os.makedirs(pickle_dir, exist_ok=True) # 없으면 생성

DATA_COLLECTION_PERIOD = 1200 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')  # timedelta(days=n) 달력일 기준


# chickchick.com에서 종목 리스트 조회
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())

for count, ticker in enumerate(tickers):
    time.sleep(2)
    stock_name = get_stock_name(tickers_dict, ticker)
    # if count % 100 == 0:
    #     print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    # 데이터 다운로드
    try:
        data = fetch_stock_data(ticker, start_date, today)
    except Exception as e:
        print(f"fetch_stock_data 실패-10: {ticker} ({stock_name}) {e}")
        continue

    if data is None or len(data) == 0:
        print(f"⚠️ 데이터 없음: {ticker} ({stock_name})")
        continue

    data = data.sort_index(ascending=True)   # 오름차순


    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        if os.path.getsize(filepath) == 0:
            raise EOFError("pickle 파일이 비어 있습니다.")

        df = pd.read_pickle(filepath)

    except (EOFError, FileNotFoundError) as e:
        print(f"pickle 파일을 읽을 수 없습니다: {filepath}")
        print(e)
        df = pd.DataFrame()

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음
        df = df.sort_index()
    else:
        df = data.copy()

    # 파일 저장 (임시 파일 생성 후 교체)
    tmp_filepath = filepath + ".tmp"

    df.to_pickle(tmp_filepath)
    os.replace(tmp_filepath, filepath)

    # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}] - {len(df)}")

end = time.time()     # 끝 시간(초)
elapsed = end - start

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 10_update_stock_data.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')
