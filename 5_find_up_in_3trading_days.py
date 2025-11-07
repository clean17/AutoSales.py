'''
3일전 10% 상승 후 3일동안 오르지 않은 종목을 찾는 스크립트
'''
print('3일전 10% 상승 후 3일동안 오르지 않은 종목을 찾는 스크립트')

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


def passes_rule(data: pd.DataFrame, close_cols=('종가','Close')) -> bool:
    """
    조건:
    1) '오늘부터 3일 전'의 일간 등락률(%) >= +10
       (즉, t-3 일의 종가가 t-4 대비 +10% 이상)
    2) 오늘(t), 1일 전(t-1), 2일 전(t-2) 각각의 일간 등락률이 모두
       +3% 미만이고 -3% 초과 (즉, -3% < 수익률 < +3%)

    data: 날짜 오름차순으로 정렬된 OHLCV DataFrame (마지막 행이 '오늘')
    close_cols: 종가 컬럼 후보
    """
    # 종가 컬럼 결정
    col = next((c for c in close_cols if c in data.columns), None)
    if col is None:
        raise KeyError("종가 컬럼이 없습니다. ('종가' 또는 'Close' 필요)")

    # 일간 등락률(%) 계산
    close = pd.to_numeric(data[col], errors='coerce')
    ret = close.pct_change() * 100.0

    # 필요한 인덱스가 다 있는지(최소 5영업일 필요: t-4 ~ t)
    # ret.iloc[-4] → t-3의 등락률, ret.iloc[-3:-0] → t-2, t-1, t
    if len(ret) < 5 or ret.iloc[-4:-1].isna().any() or pd.isna(ret.iloc[-1]):
        return False

    # 1) 3거래일 전 +10% 이상 ########################################################
    cond_3ago_up10 = ret.iloc[-4] >= 8.0

    # 2) 최근 3일 모두 -3% < 수익률 < +3%
    last3 = ret.iloc[-3:]                     # [t-2, t-1, t]
    cond_last3_band = ((last3 > -3.0) & (last3 < 3.0)).all()

    return bool(cond_3ago_up10 and cond_last3_band)



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['114190']  # 디버깅

AVERAGE_TRADING_VALUE = 2_000_000_000 # 평균거래대금 20억

idx = 0
while idx <= 0:   # -10까지 포함해서 돌리고, 다음 증가 전에 멈춤
    idx += 1

    for count, ticker in enumerate(tickers):
        condition_passed = True
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)

#         df = df[:-1]  # 디버깅
        data = df
#         print(data[-1:])
#         print(data)

        # 한국/영문 칼럼 자동 식별
        col_o = _col(df, '시가',   'Open')
        col_h = _col(df, '고가',   'High')
        col_l = _col(df, '저가',   'Low')
        col_c = _col(df, '종가',   'Close')
        col_v = _col(df, '거래량', 'Volume')

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

        # 최근 3거래일 거래대금이 기준치 이하면 패스
        recent_5data = data.tail(3)
        recent_5trading_value = recent_5data[col_v] * recent_5data[col_c]
        recent_average_trading_value = recent_5trading_value.mean()
        if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
            formatted_recent_value = f"{recent_average_trading_value / 100_000_000:.0f}억"
            # print(f"                                                        최근 3거래일 평균 거래대금({formatted_recent_value})이 부족 → pass")
            continue

        ########################################################################

        passed = passes_rule(data)
        if not passed:
            continue

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
        output_dir = 'D:\\333'
        os.makedirs(output_dir, exist_ok=True)

        final_file_name = f'{today} {stock_name} [{ticker}].png'
        print(final_file_name)
        final_file_path = os.path.join(output_dir, final_file_name)
        plt.savefig(final_file_path)
        plt.close()



#         try:
#             requests.post(
#                 'https://chickchick.shop/func/stocks/interest',
#                 json={
#                     "nation": "kor",
#                     "stock_code": str(ticker),
#                     "stock_name": str(stock_name),
#                     "pred_price_change_3d_pct": "",
#                     "yesterday_close": str(yesterday_close),
#                     "current_price": str(today_close),
#                     "today_price_change_pct": str(change_pct_today),
#                     "avg5d_trading_value": str(avg5),
#                     "current_trading_value": str(today_val),
#                     "trading_value_change_pct": str(ratio),
#                     "image_url": str(final_file_name),
#                     "market_value": str(market_value),
#                     "category": str(category),
#                     "target": "low",
#                 },
#                 timeout=5
#             )
#         except Exception as e:
#             # logging.warning(f"progress-update 요청 실패: {e}")
#             print(f"progress-update 요청 실패-4(4): {e}")
#             pass  # 오류