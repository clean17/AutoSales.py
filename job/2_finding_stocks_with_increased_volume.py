'''
거래대금 증가 종목 탐색
지난 5 거래일에 비해 오늘 거래대금이 x배 이상 상승한 종목 찾기
'''

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
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

from utils import fetch_stock_data, get_kor_ticker_list, get_kor_ticker_dict_list, add_technical_features, \
    plot_candles_weekly, plot_candles_daily, drop_trading_halt_rows, drop_sparse_columns, get_stock_name, \
    is_korean_stock_business_day, safe_rate, get_ticker_info

if not is_korean_stock_business_day(verbose=False):
    # print("한국증시 영업일이 아니므로 실행하지 않습니다.")
    sys.exit(0)


start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 2_finding_stocks_with_increased_volume.py...')

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
TRADING_VALUE = 4_000_000_000 # 거래대금 40억
TODAY_RATE_OF_INCREASE = 3

# tickers = get_kor_ticker_list()
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['089030']
# ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

# 결과를 저장할 배열
results = []
results2 = []

for count, ticker in enumerate(tickers):
    condition_passed = True   # 최근 10일간 변동폭이 6% 박스권 안쪽
    condition_passed2 = True  # 최근 5거래일 대비 오늘 거래대금이 5배 초과하지 않음
    time.sleep(0.1)  # x00ms 대기
    # stock_name = get_stock_name(tickers_dict, ticker)
    info = get_ticker_info(ticker, tickers_dict)

    ticker = info["ticker"]
    stock_name = info["stock_name"]
    stock_market = info["stock_market"]
    sector_code = info["sector_code"]


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_yesterday, today)

        if len(data) == 0:
            continue

        data = data.sort_index(ascending=True)   # 오름차순
        date_str = data.index[-1].strftime("%Y%m%d")  # 인덱스를 왜 맨날 바꾸는거야..?

        if date_str != today:
            continue

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음

    # 파일 저장
    df.to_pickle(filepath)
    data = df

    ########################################################################

    closes = data['종가'].values
    last_close = closes[-1]

    trading_value = data['거래량'] * data['종가']
    # 금일 거래대금 40억 이하 패스
    if trading_value.iloc[-1] < TRADING_VALUE:
        continue

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 50:
        continue

    # 700원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row['종가'] < 700:
        continue

    # 이상치 제거
    data, removed_idx = drop_trading_halt_rows(data)

    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    data, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)

    # drop 이후 다시 생성
    data = add_technical_features(data)


    # 5일, 20일 이동평균선 없으면 패스
    REQUIRED_COLS = ["MA5", "MA20"]

    for col in REQUIRED_COLS:
        if col not in data.columns:
            continue

    # 5일선이 너무 하락하면
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    min_slope = -3
    if ma5_today < data['MA20'].iloc[-1] and change_rate * 100 < min_slope:
        continue

    ########################################################################
    # ======== 조건 체크 시작 ========

    # ─────────────────────────────────────────────────────────────
    # 1) 오늘 등락률 3% 안되면 pass
    # ─────────────────────────────────────────────────────────────
    # 오늘 등락률(어제→오늘)
    today_close = closes[-1]
    yesterday_close = closes[-2]
    today_price_change_pct = round(safe_rate(today_close, yesterday_close), 2)

    # 오늘 상승률이 X% 가 안되면 제외
    if today_price_change_pct < TODAY_RATE_OF_INCREASE:
        continue


    # ─────────────────────────────────────────────────────────────
    # 2) 시가 총액 700억 이하 패스
    # ─────────────────────────────────────────────────────────────
    try:
        res = requests.post(
            'https://chickchick.kr/stocks/info',
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
            'https://chickchick.kr/stocks/overview',
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

        # 시가총액이 700억보다 작으면 패스
        if (market_value < 70_000_000_000):
            continue

    except Exception as e:
        print(f"overview 요청 실패-2: {e} {product_code}")
        pass  # 오류

    # ─────────────────────────────────────────────────────────────
    # 3) 10일 동안 박스권아니면 제외
    # ─────────────────────────────────────────────────────────────
    # "어제부터 10일 전"의 박스권 체크
    box_closes = closes[-11:-1]  # 10일 전 ~ 오늘 이전 (10개)
    max_close = box_closes.max()
    min_close = box_closes.min()
    range_pct = safe_rate(max_close, min_close)

    # 최근 10일간 변동폭이 6% 이상이면 박스권 탈출 → 관심종목 제외
    if range_pct >= 6:
        condition_passed = False


    # ─────────────────────────────────────────────────────────────
    # 4) 한국거래소(KRX)에서 공매도 과열 종목을 지정하는 조건
    #    최근 5거래일 평균 거래대금(오늘 제외)을 기준
    #    오늘 거래대금이 그 평균 대비 코스피는 6배 이상, 코스닥은 5배 이상이면 과열로 판단
    # ─────────────────────────────────────────────────────────────
    # 지난 5거래일 평균 거래대금(오늘 제외: -6:-1), 오늘값: -1
    avg5 = trading_value.iloc[-6:-1].mean()
    # 최근 5일 거래대금이 없으면 한달 평균
    if avg5 == 0.0:
        avg5 = trading_value.iloc[-21:-1].mean()
    # 오늘 거래대금
    today_val = trading_value.iloc[-1]

    # 0 나눗셈 방지 및 조건 체크
    if avg5 > 0 and np.isfinite(avg5):
        if stock_market == 'kospi' and today_val >= 6 * avg5:
           condition_passed2 = False
        if stock_market == 'kosdaq' and today_val >= 5 * avg5:
            condition_passed2 = False

    if condition_passed is False and condition_passed2 is False:
        continue

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
    year = today[:4]
    month = today[4:6]
    day = today[6:8]

    output_dir = f'F:\\interest_stocks\\{year}\\{month}\\{day}'
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
            'https://chickchick.kr/stocks/amount',
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
    5% 이상 상승 + 10일동안 박스권 
    """
    # DB 등록
    if condition_passed:
        # 부합하면 결과에 저장 (상승률, 종목명, 코드)}
        results.append((today_price_change_pct, stock_name, ticker, today_close, yesterday_close))

        try:
            requests.post(
                'https://chickchick.kr/stocks/interest/insert',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "stock_name": str(stock_name),
                    "pred_price_change_3d_pct": "",
                    "yesterday_close": str(yesterday_close),
                    "current_price": str(today_close),
                    "today_price_change_pct": str(today_price_change_pct),
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
    5% 이상 상승 + 거래대금 증가 x배 이하(과열 제외)  
    """
    if condition_passed2:
        results2.append((ratio, stock_name, ticker, float(today_val), float(avg5)))

        try:
            requests.post(
                'https://chickchick.kr/stocks/interest/insert',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "stock_name": str(stock_name),
                    "pred_price_change_3d_pct": "",
                    "yesterday_close": str(yesterday_close),
                    "current_price": str(today_close),
                    "today_price_change_pct": str(today_price_change_pct),
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

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

# if elapsed > 20:
#     print(f"2_finding_stocks_with_increased_volume.py - 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 2_finding_stocks_with_increased_volume.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')