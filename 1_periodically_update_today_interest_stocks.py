'''
관심 종목 5분 마다 데이터 갱신
'''

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import time

start = time.time()   # 시작 시간(초)
nowTime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
print(f'1_periodically_update_today_interest_stocks.py...')

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, get_kor_interest_ticker_dick_list, add_technical_features, \
    plot_candles_weekly, plot_candles_daily, drop_sparse_columns, drop_trading_halt_rows



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers_dict = get_kor_interest_ticker_dick_list()
tickers = list(tickers_dict.keys())


for count, ticker in enumerate(tickers):
    time.sleep(1)  # 1초 대기
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_yesterday, today)

    # 중복 제거 & 새로운 날짜만 추가
    # if not df.empty:
    #     # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
    #     new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
    #     df = pd.concat([df, new_rows])
    # else:
    #     df = data

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음

    data = df

    ########################################################################

    closes = data['종가'].values
    trading_value = data['거래량'] * data['종가']

    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    data = cleaned

    data, removed_idx = drop_trading_halt_rows(data)


    ########################################################################
    # ======== 조건 체크 시작 ========

    # ─────────────────────────────────────────────────────────────
    # 1) 오늘 등락률 조건
    # ─────────────────────────────────────────────────────────────
    # 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    # print('today', today_close)
    yesterday_close = closes[-2]
    # print('yesterday', yesterday_close)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    if change_pct_today < 5:
        # 주기적으로 데이터를 갱신하기 위한 스크립트는 체크하지 않는다
        pass
        # continue  # 오늘 10% 미만 상승이면 제외




    # ─────────────────────────────────────────────────────────────
    # 그래프 생성
    # ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    plot_candles_daily(data, show_months=6, title=f'{today} {stock_name} [{ticker}] Daily Chart',
                       ax_price=ax_d_price, ax_volume=ax_d_vol)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol)

    plt.tight_layout()
    # plt.show()

    # 파일 저장 (옵션)
    output_dir = 'D:\\interest_stocks'
    os.makedirs(output_dir, exist_ok=True)

    final_file_name = f'{today} {stock_name} [{ticker}].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()



    avg5 = trading_value.iloc[-6:-1].mean()
    # 최근 5일 거래대금이 없으면 한달 평균
    if avg5 == 0.0:
        avg5 = trading_value.iloc[-21:-1].mean()
    today_val = trading_value.iloc[-1]

    ratio = today_val / avg5 * 100
    ratio = round(ratio, 2)
    # 부합하면 결과에 저장 (상승률, 종목명, 코드)}
    change_pct_today = round(change_pct_today, 2)

    try:
        res = requests.post(
            'https://chickchick.shop/func/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=10
        )
        json_data = res.json()
        # json_data["result"][0]
        product_code = json_data["result"][0]["data"]["items"][0]["productCode"]

    except Exception as e:
        print(f"info 요청 실패-1: {str(ticker)} {stock_name} {e}")
        pass  # 오류

    if product_code is not None:
        # 현재 종가 가져오기
        try:
            res = requests.post(
                'https://chickchick.shop/func/stocks/amount',
                json={
                    "product_code": str(product_code)
                },
                timeout=10
            )
            json_data = res.json()
            last_close = json_data["result"]["candles"][0]["close"]
        except Exception as e:
            print(f"progress-update 요청 실패-1-1: {e}")
            pass  # 오류

    if last_close is not None:
        try:
            requests.post(
                'https://chickchick.shop/func/stocks/interest/insert',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "stock_name": str(stock_name),
                    "pred_price_change_3d_pct": "",
                    "yesterday_close": str(yesterday_close),
                    "current_price": str(today_close),
                    "today_price_change_pct": str(change_pct_today),
                    "avg5d_trading_value": str(avg5),
                    "current_trading_value": str(today_val),
                    "trading_value_change_pct": str(ratio),
                    "image_url": str(final_file_name),
                    "market_value": "",
                    "last_close": str(last_close),
                },
                timeout=10
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패-1-2: {e}")
            pass  # 오류

end = time.time()     # 끝 시간(초)
elapsed = end - start
# print(f"총 소요 시간: {elapsed:.2f}초")