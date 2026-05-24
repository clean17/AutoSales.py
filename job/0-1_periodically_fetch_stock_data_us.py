'''
장중에 주기적으로 주식 데이터를 갱신하는 스크립트..
'''

import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import pytz
from datetime import datetime, timedelta
from pathlib import Path
import time

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_usd_krw_rate, get_nasdaq_symbols, fetch_stock_data_us


start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 0_periodically_fetch_stock_data.py...')

DATA_COLLECTION_PERIOD = 1200

exchangeRate = get_usd_krw_rate()
if exchangeRate is None:
    print('#######################   exchangeRate is None   #######################')
else:
    print(f'#######################   exchangeRate is {exchangeRate}   #######################')

# 미국 동부 시간대 설정
now_us = datetime.now(pytz.timezone('America/New_York'))
# 현재 시간 출력
print("미국 동부 시간 기준 현재 시각:", now_us.strftime('%Y-%m-%d %H:%M:%S'))
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
start_five_date_us = (now_us - timedelta(days=5)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, '../pickle_us')

end_date = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')


# tickers = extract_stock_code_from_filenames(output_dir)
tickers = get_nasdaq_symbols()
# tickers = ['MNKD']


def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en


for count, ticker in enumerate(tickers):
    # print(f"Processing {count+1}/{len(tickers)} : {ticker}")
    if count % 100 == 0:
        print(f"Processing {count+1}/{len(tickers)} : {ticker}")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data_us(ticker, start_five_date_us, end_date)
    else:
        df = pd.DataFrame()
        data = fetch_stock_data_us(ticker, start_date_us, end_date)


    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음
    else:
        df = data.copy()


    # 파일 저장
    df.to_pickle(filepath)

end = time.time()     # 끝 시간(초)
elapsed = end - start

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 0-1_periodically_fetch_stock_data_us.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')