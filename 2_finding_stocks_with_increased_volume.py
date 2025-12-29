'''
거래대금 증가 종목 탐색
지난 5 거래일에 비해 오늘 거래대금이 x배 이상 상승한 종목 찾기
'''

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unicodedata
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import time

start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'🕒 {nowTime} - running 2_finding_stocks_with_increased_volume.py...')

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, get_kor_ticker_list, get_kor_ticker_dict_list, add_technical_features, \
    plot_candles_weekly, plot_candles_daily, drop_trading_halt_rows, drop_sparse_columns



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

# tickers = get_kor_ticker_list()
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['089030']
# ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

# 결과를 저장할 배열
results = []
results2 = []

for count, ticker in enumerate(tickers):
    condition_passed = True
    condition_passed2 = True
    time.sleep(0.1)  # x00ms 대기
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    # if count % 100 == 0:
        # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_yesterday, today)

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음

    # 파일 저장
    df.to_pickle(filepath)
    data = df
    # print(data)

    ########################################################################

    closes = data['종가'].values
    # print(closes)
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

    # 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    if len(cols_to_drop) > 0:
        # print("    Drop candidates:", cols_to_drop)
        pass
    data = cleaned

    data, removed_idx = drop_trading_halt_rows(data)
    if len(removed_idx) > 0:
        # print(f"                                                        거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")
        pass

    if 'MA5' not in data.columns or 'MA20' not in data.columns:
        # print(f"                                                        이동평균선이 존재하지 않음 → pass")
        continue

    # 5일선이 너무 하락하면
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    min_slope = -3
    if ma5_today < data['MA20'].iloc[-1] and change_rate * 100 < min_slope:
        # print(f"                                                        5일선이 20일선 보다 낮으면서 {min_slope}기울기보다 낮게 하락중[{change_rate * 100:.2f}] → pass")
        continue
        # pass

    ########################################################################
    # ======== 조건 체크 시작 ========

    # ─────────────────────────────────────────────────────────────
    # 1) 10일 동안 박스권 >>> 오늘 급등 찾기
    # ─────────────────────────────────────────────────────────────
    # "어제부터 10일 전"의 박스권 체크
    box_closes = closes[-11:-1]  # 10일 전 ~ 오늘 이전 (10개)
    # print('box', box_closes)
    max_close = box_closes.max()
    # print('max', max_close)
    min_close = box_closes.min()
    # print('min', min_close)
    range_pct = (max_close - min_close) / min_close * 100

    # 10일 동안 5% 이상 변화가 없다 -> 박스권으로 간주
    if range_pct >= 5:
        condition_passed = False
        # continue  # 4% 이상 움직이면 박스권 X

    # 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    # print('today', today_close)
    yesterday_close = closes[-2]
    # print('yesterday', yesterday_close)
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100

    # 오늘 상승률이 X% 가 안되면 제외
    if change_pct_today < 5:
        condition_passed = False
        condition_passed2 = False
        # continue  # 오늘 10% 미만 상승이면 제외


    # ─────────────────────────────────────────────────────────────
    # 2) 시가 총액 500억 이하 패스
    # ─────────────────────────────────────────────────────────────
    try:
        res = requests.post(
            'https://chickchick.shop/func/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=10
        )
        json_data = res.json()
        result = json_data["result"]

        # 거래정지는 데이터를 주지 않는다
        if len(result) == 0:
            continue

        product_code = result[0]["data"]["items"][0]["productCode"]

    except Exception as e:
        print(f"info 요청 실패-2: (코드: {str(ticker)}, 종목명: {stock_name}) {e}")
        pass  # 오류

    try:
        res2 = requests.post(
            'https://chickchick.shop/func/stocks/overview',
            json={"product_code": str(product_code)},
            timeout=10
        )
        data2 = res2.json()
        # if data2 is not None:
        market_value = data2["result"]["marketValueKrw"]
        company_code = data2["result"]["company"]["code"]

        if market_value is None:
            print(f"overview marketValueKrw is None: {product_code}")
            continue

        # 시가총액이 500억보다 작으면 패스
        if (market_value < 50_000_000_000):
            continue

    except Exception as e:
        print(f"overview 요청 실패-2: {e} {product_code}")
        pass  # 오류


    # ─────────────────────────────────────────────────────────────
    # 3) 5일 평균 거래대금 * X < 오늘 거래대금 찾기
    # ─────────────────────────────────────────────────────────────
    # 지난 5거래일 평균 거래대금(오늘 제외: -6:-1), 오늘값: -1
    avg5 = trading_value.iloc[-6:-1].mean()
    # 최근 5일 거래대금이 없으면 한달 평균
    if avg5 == 0.0:
        avg5 = trading_value.iloc[-21:-1].mean()
    # print('avg', avg5)
    today_val = trading_value.iloc[-1]
    # print('today', today_val)

    # 거래대금 x배 보다 크면 과열 > 제외
    TARGET_VALUE = 6
    # 0 나눗셈 방지 및 조건 체크
    if avg5 > 0 and np.isfinite(avg5) and today_val >= TARGET_VALUE * avg5:
        condition_passed2 = False


    # ─────────────────────────────────────────────────────────────
    # 그래프 생성
    # ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    plot_candles_daily(data, show_months=4, title=f'{today} {stock_name} [{ticker}] Daily Chart',
                       ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

    plt.tight_layout()
    # plt.show()

    # 파일 저장 (옵션)
    output_dir = 'D:\\interest_stocks'
    os.makedirs(output_dir, exist_ok=True)

    final_file_name = f'{today} {stock_name} [{ticker}].webp'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path, format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()


    ratio = today_val / avg5 * 100
    ratio = round(ratio, 2)

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
        print(f"progress-update 요청 실패-2-1: {e}")
        pass  # 오류

    """
    5% 이상 상승 + 10일동안 4-5% 박스권 
    """
    # DB 등록
    if condition_passed:
        # 부합하면 결과에 저장 (상승률, 종목명, 코드)}
        change_pct_today = round(change_pct_today, 2)
        results.append((change_pct_today, stock_name, ticker, today_close, yesterday_close))

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
                    "graph_file": str(final_file_name),
                    "market_value": "0" if market_value is None else str(market_value),
                    "last_close": str(last_close),
                },
                timeout=10
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패-2-2: {e}")
            pass  # 오류

    """
    5% 이상 상승 + 거래대금 증가 5-6배 이하(과열 제외)  
    """
    if condition_passed2:
        results2.append((ratio, stock_name, ticker, float(today_val), float(avg5)))

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
                    "graph_file": str(final_file_name),
                    "market_value": str(market_value),
                    "last_close": str(last_close),
                },
                timeout=10
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패-2-3: {e}")
            pass  # 오류



# ─────────────────────────────────────────────────────────────
# 콘솔 출력
# ─────────────────────────────────────────────────────────────
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


if len(results2) > 0:
    # 내림차순 정렬 (상승률 기준)
    results2.sort(reverse=True, key=lambda x: x[0])


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
    max_name_vis_len = max(visual_width(name) for _, name, _, _, _ in results2)

    # 시각적 폭에 맞춰 공백 패딩
    def pad_visual(text, target_width):
        gap = target_width - visual_width(text)
        return text + ' ' * gap

    for ratio, stock_name, ticker, today_val, avg5 in results2:
        print(f"==== {pad_visual(stock_name, max_name_vis_len)} [{ticker}]  {avg5/100_000_000:.1f}억 >>> {today_val/100_000_000:.2f}억, 거래대금 상승률 : {ratio:,.1f}% ====")

end = time.time()     # 끝 시간(초)
elapsed = end - start
print(f"2_finding_stocks_with_increased_volume 총 소요 시간: {elapsed:.2f}초")