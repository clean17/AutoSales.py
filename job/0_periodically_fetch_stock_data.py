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

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, get_kor_ticker_dict_list, get_stock_name, is_korean_stock_business_day, \
    safe_replace_pickle

if not is_korean_stock_business_day(verbose=False):
    # print("한국증시 영업일이 아니므로 실행하지 않습니다.")
    sys.exit(0)


start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 0_periodically_fetch_stock_data.py...')

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())


for count, ticker in enumerate(tickers):
    time.sleep(0.1)  # x00ms 대기
    stock_name = get_stock_name(tickers_dict, ticker)
    # if count % 100 == 0:
    #     print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        if os.path.getsize(filepath) == 0:
            raise EOFError("⚠️ pickle 파일이 비어 있습니다.")

        df = pd.read_pickle(filepath)

    except (EOFError, FileNotFoundError) as e:
        print(f"⚠️ pickle 파일을 읽을 수 없습니다-0: {filepath}")
        print(e)
        df = pd.DataFrame()


    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # 1일 데이터 받아서 병합
        try:
            data = fetch_stock_data(ticker, start_yesterday, today)
        except Exception as e:
            print(f"⚠️ fetch_stock_data 실패-0: {ticker} ({stock_name}) {e}")
            continue

        if data is None or data.empty:
            print(f"⚠️ 데이터 없음-0: {ticker} ({stock_name})")
            continue

        data = data.sort_index(ascending=True)   # 오름차순
        # if count == 1:
        #     print(data)
        #     print('date_str', date_str)
        #     print('today', today)

        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음

        # 파일 저장 (임시 파일 생성 후 교체)
        safe_replace_pickle(df, filepath)

end = time.time()     # 끝 시간(초)
elapsed = end - start

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 0_periodically_fetch_stock_data.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')