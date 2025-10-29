'''
저점을 찾는 스크립트
'''

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import requests

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop


def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

def weekly_check(data: pd.DataFrame):
    # 인덱스가 날짜/시간이어야 함
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        data = data.copy()
        data.index = pd.to_datetime(data.index)

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    # 주봉 리샘플 (월~금 장 기준이면 W-FRI 권장)
    weekly = data.resample('W-FRI').agg({
        col_o: 'first',
        col_h: 'max',
        col_l: 'min',
        col_c: 'last',
        col_v: 'sum'
    }).dropna(subset=[col_c])  # 종가 없는 주 제거

    # 직전 2주 추출
    prev_close = weekly.iloc[-2][col_c]
    this_close = weekly.iloc[-1][col_c] # 마지막 주 종가

    past_min = weekly.iloc[:-1][col_c].min()  # 이번 주 제외 과거 최저
    first = weekly.iloc[0][col_c]             # 첫번째 주 종가

    # 20% 이상 하락? (현재가가 과거최저의 80% 이하)
    is_drop_20 = this_close <= first * 0.8
    pct_from_min = this_close / first - 1.0  # 이번 주 종가(this_close)가 첫 번째 주 종가(first) 대비 몇 % 변했는지

    pct = (this_close / prev_close) - 1  # 이번주 대비 전주 증감률
    is_higher = this_close > prev_close
    is_drop_over_3 = pct < -0.005   # -0.5% 보다 더 하락했는가

    return {
        "ok": True,
        "this_week_close": float(this_close),
        "last_week_close": float(prev_close),
        "pct_change": float(pct),      # 예: -0.0312 == -3.12%
        "is_higher_than_last_week": bool(is_higher),
        "is_drop_more_than_minus3pct": bool(is_drop_over_3),
        "weekly": weekly,
        "past_min_close": float(first),
        "first_close": float(first),
        "pct_vs_past_min": float(pct_from_min * 100),  # 예: -0.22 == -22% 하락
        "is_drop_more_than_20pct": bool(is_drop_20),
    }



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['419530', '219550', '223310', '007110', '047770', '083660', '001515', '004835', '145210', '217330', '322780', '042660', '083650', '017510', '052770', '131400', '006490', '254120', '114190', '044490', '393890', '396300', '086520', '418550', '002710', '121600', '020150', '069920', '137400', '043100', '002020', '317330', '383310', '452400', '234920', '018880', '417010', '340930']

# SPLIT_DATE = -2
# SPLIT_DATE = 0

idx = 0
while idx <= 0:   # -10까지 포함해서 돌리고, 다음 증가 전에 멈춤
    idx += 1

# while SPLIT_DATE <= 21:   # -10까지 포함해서 돌리고, 다음 증가 전에 멈춤
#     SPLIT_DATE += 1

    for count, ticker in enumerate(tickers):
        condition_passed = True
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)

        data = df
        # print(data[-1:])

        # # 검증용
        # origin = data.copy()
        # data = data[:SPLIT_DATE]

        if count == 0:
            # print(data)
            today = data.index[-1].strftime("%Y%m%d")
            print('\n─────────────────────────────────────────────────────────────')
            print(data.index[-1].date())
            print('─────────────────────────────────────────────────────────────')

        ########################################################################

        closes = data['종가'].values
        trading_value = data['거래량'] * data['종가']


        # 데이터가 부족하면 패스
        if data.empty or len(data) < 70:
            # print(f"                                                        데이터 부족 → pass")
            continue

        # 2차 생성 feature
        data = add_technical_features(data)

        # 결측 제거
        cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
        # o_cleaned, cols_to_drop = drop_sparse_columns(origin, threshold=0.10, check_inf=True, inplace=True)
        data = cleaned
        # origin = o_cleaned

        # 거래정지/이상치 행 제거
        data, removed_idx = drop_trading_halt_rows(data)
        # origin, removed_idx = drop_trading_halt_rows(origin)


        if 'MA5' not in data.columns or 'MA20' not in data.columns:
            continue

        # 5일선은 20일선보다 낮아야 한다
        ma5_today = data['MA5'].iloc[-1]
        ma20_today = data['MA20'].iloc[-1]

        if ma5_today >= ma20_today:
            continue

        # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승
        signal = signal_any_drop(data)
        if not signal:
            continue

        # 오늘 14% 이상 오르면 패스
        if data.iloc[-1]['등락률'] > 13:
            continue


        ########################################################################

        m_data = data[-100:]

        # n = len(origin)
        # start = SPLIT_DATE
        #
        # # 음수 시작은 뒤에서부터로 해석해 양수로 변환
        # if start < 0:
        #     start = max(0, n + start)
        # nn = 10 if SPLIT_DATE <= -10 else 5
        # stop = min(n, start + nn)  # 범위 초과 방지
        # origin = origin.iloc[start:stop]
        # # print(origin) # 미래 가격 비교

        # print('m_date:', m_data.index[0].date())
        # print(origin.index[0].date()) #  2025-05-20, 대략 5달

        m_closes = m_data['종가']
        # o_closes = origin['종가']
        m_max = m_closes.max()
        m_min = m_closes.min()
        m_current = m_closes[-1]
        # o_max = o_closes.max() # m_current의 다음 기간 최대

        # print('MAX:',m_max)
        # print('MIN:', m_min)
        # print('TODAY:', m_current)

        m_chg_rate=(m_max-m_min)/m_min*100          # 최근 5달 동안의 변동률
        # m_chg_rate2=(o_max-m_current)/m_current*100 # 미래가격이 현재가 대비 얼마나 올랐어?
        m_chg_rate3=(m_current-m_max)/m_max*100 # 최근 5달 최대 대비 하락률 계산

        # 최근 변동률 최소 기준: 횡보는 패스 (보조)
        if m_chg_rate < 20:
            continue

        # 미래가 10% 이상 상승(검증용)
        # if m_chg_rate2 <= 9:
        #     continue


        result = weekly_check(m_data)
        if result["ok"]:
            # print(f"이번주 종가: {result['this_week_close']:.2f}")
            # print(f"지난주 종가: {result['last_week_close']:.2f}")
            # print(f"지난주 대비 변동률: {result['pct_change']*100:.2f}%")
            # print("이번주가 더 높음:", result["is_higher_than_last_week"])
            # print("-0.5%보다 더 하락:", result["is_drop_more_than_minus3pct"])
            # print(f"5개월 주봉 변동률: {result['pct_vs_past_min']:.1f}")
            # print(f"5개월 전 주봉보다 하락: {result['is_drop_more_than_20pct']}")

            # 저번주 대비 이번주 증감률 -0.5보다 낮으면 패스
            if result["is_drop_more_than_minus3pct"]:
                continue

            # 지난주 대비 주봉 종가가 15% 이상 상승하면 패스
            if result['pct_change'] * 100 > 15:
                continue

            # 직전 날까지의 마지막 3일 거래대금 평균
            today_tr_val = trading_value.iloc[-1]
            mean_prev3 = trading_value.iloc[:-1].tail(3).mean()
            chg_tr_val = (today_tr_val-mean_prev3)/mean_prev3*100

            # 3거래일 평균 거래대금 5억보다 작으면 패스
            if mean_prev3.round(1) / 100_000_000 < 5:
                continue

            # 5개월 주봉 변동률: 너무 하락한것 제외 (목 돌아감), (보조)
            if result['pct_vs_past_min'] < -20:
                continue

            # 거래대금 변동률 50% 이상 (보조)
            if chg_tr_val < 50:
                continue



            print(f"\nProcessing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")
            print(f"  직전 3일 거래대금: {mean_prev3.round(1) / 100_000_000:.0f}억")
            print(f"  오늘 거래대금: {today_tr_val.round(1) / 100_000_000:.0f}억")
            print(f"  거래대금 변동률: {chg_tr_val:.1f}%")
            print(f'  5개월 min_max 종가 변동률: {m_chg_rate.round(1)}%', )
            # print(f'  검증 상승률: {m_chg_rate2.round(1)}%')
            print(f"  5개월 주봉 변동률: {result['pct_vs_past_min']:.1f}")
            print(f"  5개월 전 주봉보다 하락: {result['is_drop_more_than_20pct']}")
            print(f"  5개월 최대 대비 하락률: {m_chg_rate3:.2f}%")
            print(f"  지난주 대비 변동률: {result['pct_change']*100:.1f}%")
            print(f"  오늘 등락률: {data.iloc[-1]['등락률']:.2f}%")


            today_close = closes[-1]
            yesterday_close = closes[-2]
            change_pct_today = (today_close - yesterday_close) / yesterday_close * 100
            change_pct_today = round(change_pct_today, 2)
            avg5 = trading_value.iloc[-6:-1].mean()
            today_val = trading_value.iloc[-1]
            ratio = today_val / avg5 * 100
            ratio = round(ratio, 2)

            try:
                res = requests.post(
                    'https://chickchick.shop/func/stocks/info',
                    json={"stock_name": str(ticker)},
                    timeout=10
                )
                json_data = res.json()
                product_code = json_data["result"][0]["data"]["items"][0]["productCode"]
            except Exception as e:
                print(f"info 요청 실패-4(1): {e}")
                pass  # 오류

            try:
                res2 = requests.post(
                    'https://chickchick.shop/func/stocks/overview',
                    json={"product_code": str(product_code)},
                    timeout=10
                )
                data2 = res2.json()
                market_value = data2["result"]["marketValueKrw"]
                company_code = data2["result"]["company"]["code"]
            except Exception as e:
                print(f"overview 요청 실패-4(2): {e}")
                pass  # 오류

            try:
                res = requests.post(
                    'https://chickchick.shop/func/stocks/company',
                    json={"company_code": str(company_code)},
                    timeout=15
                )
                json_data = res.json()
                category = json_data["result"]["majorList"][0]["title"]
            except Exception as e:
                print(f"/func/stocks/company 요청 실패-4(3): {e}")
                pass  # 오류


        ########################################################################

        # 그래프 생성
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
        output_dir = 'D:\\5below20'
        os.makedirs(output_dir, exist_ok=True)

        final_file_name = f'{today} {stock_name} [{ticker}].png'
        final_file_path = os.path.join(output_dir, final_file_name)
        plt.savefig(final_file_path)
        plt.close()



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
                    "category": str(category),
                    "target": "low",
                },
                timeout=5
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패-4(4): {e}")
            pass  # 오류