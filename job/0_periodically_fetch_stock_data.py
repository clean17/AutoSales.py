'''
장중에 주기적으로 주식 데이터를 갱신하는 스크립트
'''

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 0_periodically_fetch_stock_data.py...')

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, get_kor_ticker_dict_list


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, '../pickle')

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())


for count, ticker in enumerate(tickers):
    time.sleep(0.1)  # x00ms 대기
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    # if count % 100 == 0:
    #     print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_yesterday, today)

        if len(data) == 0:
            continue

        data = data.sort_index(ascending=True)   # 오름차순
        date_str = data.index[-1].strftime("%Y%m%d")  # 인덱스를 왜 맨날 바꾸는거야..?
        # if count == 1:
        #     print(data)
        #     print('date_str', date_str)
        #     print('today', today)

        if date_str != today:
            continue

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음

    # 파일 저장
    df.to_pickle(filepath)

end = time.time()     # 끝 시간(초)
elapsed = end - start

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 0_periodically_fetch_stock_data.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')