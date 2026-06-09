'''
관심 종목 5분 마다 데이터 갱신
'''

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
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

from utils import fetch_stock_data, get_kor_interest_ticker_dick_list, add_technical_features, get_stock_name, \
    plot_candles_weekly, plot_candles_daily, drop_sparse_columns, drop_trading_halt_rows, \
    get_today_kor_low_ticker_dick_list, is_korean_stock_business_day, update_today_ohlcv_from_amount, \
    get_ticker_info, safe_replace_pickle

if not is_korean_stock_business_day(verbose=False):
    # print("한국증시 영업일이 아니므로 실행하지 않습니다.")
    sys.exit(0)


start = time.time()   # 시작 시간(초)
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - 🕒 running 1_periodically_update_today_interest_stocks.py...')


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")

run_today = datetime.today().strftime('%Y%m%d')
start_yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

tickers_dict = get_kor_interest_ticker_dick_list()
low_tickers_dict = get_today_kor_low_ticker_dick_list()
tickers_dict.update(low_tickers_dict)
tickers = list(tickers_dict.keys())


for count, ticker in enumerate(tickers):
    time.sleep(0.02)
    # stock_name = get_stock_name(tickers_dict, ticker)
    info = get_ticker_info(ticker, tickers_dict)

    ticker = info["ticker"]
    stock_name = info["stock_name"]
    stock_market = info["stock_market"]
    sector_code = info["sector_code"]
    product_code = info["product_code"]


    # 데이터가 없으면 1년 데이터 요청, 있으면 1일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        if os.path.getsize(filepath) == 0:
            raise EOFError("⚠️ pickle 파일이 비어 있습니다.")

        df = pd.read_pickle(filepath)

    except (EOFError, FileNotFoundError) as e:
        print(f"⚠️ pickle 파일을 읽을 수 없습니다-1: {filepath}")
        print(e)
        continue


    # pykrx 에서 증권사 api 호출로 변경
    # try:
    #     data = fetch_stock_data(ticker, start_yesterday, run_today)
    # except Exception as e:
    #     print(f"⚠️ fetch_stock_data 실패-1: {ticker} {stock_name} {e}")
    #     continue
    #
    #
    # if data is None or data.empty:
    #     print(f"⚠️ 신규 데이터 없음-1: {ticker} {stock_name}")
    #     if df.empty:
    #         continue
    #     data = pd.DataFrame()
    #
    #
    # # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    # if not df.empty:
    #     # df와 data를 concat 후, data 값으로 덮어쓰기
    #     df = pd.concat([df, data])
    #     df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음
    # else:
    #     df = data.copy()


    # 증권사에 현재 ohlcv 데이터 요청
    df, today_amount = update_today_ohlcv_from_amount(product_code, df, ticker, product_code)

    # 파일 저장 (임시 파일 생성 후 교체)
    safe_replace_pickle(df, filepath)

    ########################################################################

    # 데이터가 부족하면 패스
    if df is None or df.empty or len(df) < 50:
        continue

    # 거래정지/이상치 행 제거
    df, _ = drop_trading_halt_rows(df)

    # 2차 생성 feature
    df = add_technical_features(df)

    # 결측 제거
    df, _ = drop_sparse_columns(df, threshold=0.10, check_inf=True, inplace=True)

    # drop 이후 다시 생성
    df = add_technical_features(df)

    # 데이터가 부족하면 패스
    if df.empty or len(df) < 50:
        continue


    data = df

    closes = data['종가'].values
    trading_value = data['거래량'] * data['종가']

    last_date = data.index[-1].strftime("%Y%m%d")  # 마지막 인덱스

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
    today_price_change_pct = round(float(data["등락률"].iloc[-1]), 2)


    # ─────────────────────────────────────────────────────────────
    # 2) 시가 총액 700억 이하 패스
    # ─────────────────────────────────────────────────────────────
    if product_code is None:
        print(f"product_code 없음으로 overview 스킵: {ticker} {stock_name}")
        continue

    market_value = 0

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
        else:
            market_value = int(market_value)

        # 시가총액이 700억보다 작으면 패스
        if (market_value < 70_000_000_000):
            continue

    except Exception as e:
        print(f"⚠️ overview 요청 실패-2: {e} {product_code}")


    # ─────────────────────────────────────────────────────────────
    # 그래프 생성
    # ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    plot_candles_daily(data, show_months=4, title=f'{last_date} {stock_name} [{ticker}] Daily Chart',
                       ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

    plt.tight_layout()
    # plt.show()

    # 파일 저장 (옵션)
    year = last_date[:4]
    month = last_date[4:6]
    day = last_date[6:8]

    output_dir = f'F:\\interest_stocks\\{year}\\{month}\\{day}'
    os.makedirs(output_dir, exist_ok=True)

    final_file_name = f'{last_date} {stock_name} [{ticker}].webp'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path, format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()


    today_val = today_amount if today_amount is not None else trading_value.iloc[-1]
    avg5 = trading_value.iloc[-6:-1].mean()
    # 최근 5일 거래대금이 없으면 한달 평균
    if avg5 <= 0 or not np.isfinite(avg5):
        avg5 = trading_value.iloc[-21:-1].mean()
    if avg5 <= 0 or not np.isfinite(avg5):
        trading_value_change_pct = 100
    else:
        trading_value_change_pct = today_val / avg5 * 100
        trading_value_change_pct = round(trading_value_change_pct, 2)

    avg5_trv = int(avg5) if np.isfinite(avg5) else 0

    try:
        requests.post(
            'https://chickchick.kr/stocks/interest/insert',
            json={
                "nation": "kor",
                "stock_code": str(ticker),
                "stock_name": str(stock_name),
                "pred_price_change_3d_pct": "",
                "yesterday_close": str(yesterday_close),
                "last_close": str(today_close),
                "today_price_change_pct": str(today_price_change_pct),
                "avg5d_trading_value": str(avg5_trv),
                "current_trading_value": str(today_val),
                "trading_value_change_pct": str(trading_value_change_pct),
                "graph_file": str(final_file_name),
                "market_value": str(market_value),
            },
            timeout=10
        )
    except Exception as e:
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"⚠️ progress-update 요청 실패-1-2: {e}")

    # print('last_date', last_date, ticker, stock_name)

end = time.time()     # 끝 시간(초)
elapsed = end - start

hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

# if elapsed > 20:
#     print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
print(f'{nowTime} - Complete : 1_periodically_update_today_interest_stocks.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')
