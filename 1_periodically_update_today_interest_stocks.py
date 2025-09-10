import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import time

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, get_kor_interest_ticker_dick_list, add_technical_features, \
    plot_candles_weekly, plot_candles_daily

'''
관심 종목 5분 마다 데이터 갱신
'''

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
    condition_passed = True
    condition_passed2 = True
    time.sleep(1)  # 1초 대기
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_yesterday, today)

    # 중복 제거 & 새로운 날짜만 추가
    if not df.empty:
        # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
        new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
        df = pd.concat([df, new_rows])
    else:
        df = data

    # 너무 먼 과거 데이터 버리기
    if len(df) > 280:
        df = df.iloc[-280:]

    data = df

    ########################################################################

    closes = data['종가'].values
    last_close = closes[-1]

    trading_value = data['거래량'] * data['종가']
    # 금일 거래대금 50억 이하 패스
    if trading_value.iloc[-1] < 5_000_000_000:
        continue

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 50:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 500원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row['종가'] < 500:
        # print("                                                        종가가 0이거나 500원 미만 → pass")
        continue

    # 2차 생성 feature
    data = add_technical_features(data)


    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
        # print(f"                                                        5일선이 20일선 보다 낮을 경우 → pass")
        continue

    ########################################################################
    # ======== 조건 체크 시작 ========

    """
    10일 동안 박스권 > 오늘 급등 찾기
    """
    # "어제부터 10일 전"의 박스권 체크
    box_closes = closes[-11:-1]  # 10일 전 ~ 오늘 이전 (10개)
    # print('box', box_closes)
    max_close = box_closes.max()
    # print('max', max_close)
    min_close = box_closes.min()
    # print('min', min_close)
    range_pct = (max_close - min_close) / min_close * 100

    if range_pct >= 4:
        condition_passed = False
        # continue  # 4% 이상 움직이면 박스권 X

    # 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    # print('today', today_close)
    yesterday_close = closes[-2]
    # print('yesterday', yesterday_close)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    if change_pct_today < 10:
        condition_passed = False
        condition_passed2 = False
        # continue  # 오늘 10% 미만 상승이면 제외


    # 시가 총액 500억 이하 패스
    try:
        res = requests.post(
            'https://chickchick.shop/func/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=5
        )
        json_data = res.json()
        product_code = json_data["result"][0]["data"]["items"][0]["productCode"]

        res2 = requests.post(
            'https://chickchick.shop/func/stocks/overview',
            json={"product_code": str(product_code)},
            timeout=5
        )
        data2 = res2.json()
        market_value = data2["result"]["marketValueKrw"]
        # 시가총액이 500억보다 작으면 패스
        if (market_value < 50_000_000_000):
            # condition_passed = False
            continue

    except Exception as e:
        print(f"info 요청 실패: {e}")
        pass  # 오류


    """
    5일 평균 거래대금 * 10 < 오늘 거래대금 찾기
    """
    # 지난 5거래일 평균 거래대금(오늘 제외: -6:-1), 오늘값: -1
    avg5 = trading_value.iloc[-6:-1].mean()
    # print('avg', avg5)
    today_val = trading_value.iloc[-1]
    # print('today', today_val)

    # 거래대금 x배 증가 종목 찾기
    TARGET_VALUE = 10
    # 0 나눗셈 방지 및 조건 체크
    if avg5 > 0 and np.isfinite(avg5) and today_val >= TARGET_VALUE * avg5:
        condition_passed2 = True

    # 그래프 생성
    fig = plt.figure(figsize=(16, 20), dpi=200)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    plot_candles_daily(data, show_months=6, title=f'{today} {stock_name} [{ticker}] Daily Chart',
                       ax_price=ax_d_price, ax_volume=ax_d_vol)

    plot_candles_weekly(data, show_months=12, title=f'{today} {stock_name} [{ticker}] Weekly Chart',
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


    if condition_passed:
        # 부합하면 결과에 저장 (상승률, 종목명, 코드)}
        change_pct_today = round(change_pct_today, 2)

        try:
            requests.post(
                'https://chickchick.shop/func/stocks/interest',
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
                    "market_value": str(market_value),
                },
                timeout=5
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패: {e}")
            pass  # 오류


    if condition_passed2:
        ratio = today_val / avg5 * 100
        # print('ratio', ratio)
        # 결과: (배수, 종목명, 코드, 오늘거래대금, 5일평균거래대금)

        ratio = round(ratio, 2)

        try:
            requests.post(
                'https://chickchick.shop/func/stocks/interest',
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
                    "market_value": str(market_value),
                },
                timeout=5
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패: {e}")
            pass  # 오류

