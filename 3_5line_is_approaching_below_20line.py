'''
5일선이 상승중인 ?
20일선 아래에서 상승중인 ?
반등하는 ? 주식 찾아서 그래프 생성
'''

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    near_bull_cross_signal, drop_sparse_columns



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())


# 결과를 저장할 배열
results = []


for count, ticker in enumerate(tickers):
    condition_passed = True
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)

    data = df
    # print(data)

    ########################################################################

    closes = data['종가'].values
    # print(closes)
    last_close = closes[-1]

    trading_value = data['거래량'] * data['종가']
    # 금일 거래대금 50억 이하 패스
    if trading_value.iloc[-1] < 1_000_000_000:
        # print(f"                                                        거래대금 부족 → pass")
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

    if 'MA5' not in data.columns or 'MA20' not in data.columns:
        # print(f"                                                        이동평균선이 존재하지 않음 → pass")
        continue

    # 5일선 기울기
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]
    ma20_today = data['MA20'].iloc[-1]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday
    min_slope = -1.8

    # 20일선보다 5일선이 높은데 크게 하락중이면 패스
    if ma5_today > ma20_today and change_rate * 100 < min_slope:
        # print(f"                                                        5일선이 20일선 보다 높은데 {min_slope}기울기보다 하락중[{change_rate * 100:.2f}] → pass")
        continue

    # 현재 5일선이 20일선보다 낮으면서 20일선으로 다가오지 않으면 패스
    if ma5_today < ma20_today and near_bull_cross_signal(data, lookback=5, gap_bp=0.015, min_rise_bp=0.003): # gap_bp → 0.008~0.015
        results.append((stock_name, ticker))
        # print(f"                                                        5일선이 20일선 보다 낮으면서 {min_slope}기울기보다 하락중[{change_rate * 100:.2f}] → pass")
    else:
        # print('★★★★★★★★★★★★★★★★★★★')
        continue
        # pass
    ########################################################################
    # ======== 조건 체크 시작 ========

    # if near_bull_cross_signal(data, lookback=5, gap_bp=0.004, min_rise_bp=0.002):
    #     results.append((stock_name, ticker))
    # else:
    #     continue

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
    max_name_vis_len = max(visual_width(name) for name, _ in results)

    # 시각적 폭에 맞춰 공백 패딩
    def pad_visual(text, target_width):
        gap = target_width - visual_width(text)
        return text + ' ' * gap

    for stock_name, ticker in results:
        print(f"==== {pad_visual(stock_name, max_name_vis_len)} [{ticker}] ====")

