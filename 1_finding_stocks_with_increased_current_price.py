import os, sys
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
import unicodedata
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

from utils import fetch_stock_data, get_kor_ticker_list, add_technical_features, plot_candles_weekly, plot_candles_daily

'''
급등주 탐색
최근 2주동안 횡보하다가 갑자기 폭등한 주식 찾기
'''

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

AVERAGE_TRADING_VALUE = 1_000_000_000 # 평균거래대금

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers = get_kor_ticker_list()
# tickers = ['378850']
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}


# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    time.sleep(0.2)  # 200ms 대기
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

    # # 최근 2주 거래대금이 기준치 이하면 패스
    # recent_data = data.tail(10)
    # recent_trading_value = recent_data['거래량'] * recent_data['종가']
    # recent_average_trading_value = recent_trading_value.mean()
    # if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
    #     continue


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

    # 1. "어제부터 10일 전"의 박스권 체크
    box_closes = closes[-11:-1]  # 10일 전 ~ 오늘 이전 (10개)
    # print('box', box_closes)
    max_close = box_closes.max()
    # print('max', max_close)
    min_close = box_closes.min()
    # print('min', min_close)
    range_pct = (max_close - min_close) / min_close * 100

    if range_pct >= 4:
        continue  # 4% 이상 움직이면 박스권 X

    # 2. 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    # print('today', today_close)
    yesterday_close = closes[-2]
    # print('yesterday', yesterday_close)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    if change_pct_today < 10:
        continue  # 오늘 10% 미만 상승이면 제외


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
        # 시가총액
        market_value = data2["result"]["marketValueKrw"]
        # 시가총액이 500억보다 작으면 패스
        if (market_value < 50_000_000_000):
            continue

    except Exception as e:
        print(f"info 요청 실패: {e}")
        pass  # 오류



    fig = plt.figure(figsize=(16, 20), dpi=200)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    plot_candles_daily(data, show_months=5, title=f'{today} {stock_name} [{ticker}] Daily Chart',
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


    # 부합하면 결과에 저장 (상승률, 종목명, 코드)}
    change_pct_today = round(change_pct_today, 2)
    results.append((change_pct_today, stock_name, ticker, today_close, yesterday_close))

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
                "avg5d_trading_value": "",
                "current_trading_value":"",
                "trading_value_change_pct": "",
                "image_url": str(final_file_name),
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

    for change, stock_name, ticker, current_price, yesterday_close in results:
        print(f"==== {pad_visual(stock_name, max_name_vis_len)} [{ticker}] 상승률 {change:,.2f}% ====")
