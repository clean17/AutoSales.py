import os
import numpy as np
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from utils import fetch_stock_data, add_technical_features
import unicodedata
import requests

'''
거래대금 증가 종목 탐색
지난 5 거래일에 비해 오늘 거래대금이 x배 이상 상승한 종목 찾기
'''

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

url = "https://chickchick.shop/func/stocks/kor"
res = requests.get(url)
data = res.json()
tickers = [item["stock_code"] for item in data if "stock_code" in item]
# tickers = get_kor_ticker_list()
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
        data = fetch_stock_data(ticker, start_yesterday, today)

    # 중복 제거 & 새로운 날짜만 추가
    if not df.empty:
        # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
        new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
        df = pd.concat([df, new_rows])
    else:
        df = data

    # 너무 먼 과거 데이터 버리기
    if len(df) > 270:
        df = df.iloc[-270:]

    data = df

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

    # 2차 생성 feature
    data = add_technical_features(data)

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
        # print(f"                                                        5일선이 20일선 보다 낮을 경우 → pass")
        continue

    ########################################################################
    # ======== 조건 체크 시작 ========

    closes = data['종가'].values
    trading_value = data['거래량'] * data['종가']

    # 1. 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    # print('today', today_close)
    yesterday_close = closes[-2]
    # print('yesterday', yesterday_close)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    if change_pct_today < 10:
        continue  # 오늘 10% 미만 상승이면 제외

    # 2. 거래대금
    # 지난 5거래일 평균(오늘 제외: -6:-1), 오늘값: -1
    avg5 = trading_value.iloc[-6:-1].mean()
    # print('avg', avg5)
    today_val = trading_value.iloc[-1]
    # print('today', today_val)

    # 거래대금 x배 증가 종목 찾기
    TARGET_VALUE = 10
    # 0 나눗셈 방지 및 조건 체크
    if avg5 > 0 and np.isfinite(avg5) and today_val >= TARGET_VALUE * avg5:
        ratio = today_val / avg5 * 100
        # print('ratio', ratio)
        # 결과: (배수, 종목명, 코드, 오늘거래대금, 5일평균거래대금)

        try:
            res = requests.post(
                'https://chickchick.shop/func/stocks/info',
                json={"stock_name": str(ticker)},
                timeout=5
            )
            data = res.json()
            product_code = data["result"][0]["data"]["items"][0]["productCode"]

            res2 = requests.post(
                'https://chickchick.shop/func/stocks/overview',
                json={"product_code": str(product_code)},
                timeout=5
            )
            data2 = res2.json()
            # 시가총액
            market_value = data2["result"]["marketValueKrw"]
            # 시가총액이 500억보다 작으면 패스
            if (market_value < 50_000_000_000):
                continue

        except Exception as e:
            print(f"info 요청 실패: {e}")
            pass  # 오류

        ratio = round(ratio, 2)
        results.append((ratio, stock_name, ticker, float(today_val), float(avg5)))

        try:
            requests.post(
                'https://chickchick.shop/func/stocks/interest',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "stock_name": str(stock_name),
                    "pred_price_change_3d_pct": "",
                    "yesterday_close": "",
                    "current_price": "",
                    "today_price_change_pct": "",
                    "avg5d_trading_value": str(avg5),
                    "current_trading_value": str(today_val),
                    "trading_value_change_pct": str(ratio),
                    "image_url": "",
                    "market_value": str(market_value),
                },
                timeout=5
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패: {e}")
            pass  # 오류


#######################################################################


if len(results) > 0:
    # 내림차순 정렬 (상승률 기준)
    results.sort(reverse=True, key=lambda x: x[0])


    # 글자별 시각적 너비 계산 함수 (한글/한자/일본어 2칸, 영문/숫자/특수문자 1칸)
    def visual_width(text):
        width = 0
        for c in text:
            if unicodedata.east_asian_width(c) in 'WF':  # W: Wide, F: Fullwidth
                width += 2
            else:
                width += 1
        return width

    # 시각적 폭 기준 최대값
    max_name_vis_len = max(visual_width(name) for _, name, _, _, _ in results)

    # 시각적 폭에 맞춰 공백 패딩
    def pad_visual(text, target_width):
        gap = target_width - visual_width(text)
        return text + ' ' * gap

    for ratio, stock_name, ticker, today_val, avg5 in results:
        print(f"==== {pad_visual(stock_name, max_name_vis_len)} [{ticker}]  {avg5/100_000_000:.2f}억 >>> {today_val/100_000_000:.2f}억, 거래대금 상승률 : {ratio:.2f}% ====")