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

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows


def signal_any_drop(data: pd.DataFrame,
                    days: int = 12,
                    up_thr: float = 3.0,
                    down_thr: float = -3.0) -> bool:
    """
    요구 조건:
      - 오늘 등락률(마지막 행) >= up_thr  (단위: %)
      - 어제부터 과거 days일 동안 등락률 <= down_thr 인 날이 '하루라도' 있음
      - 같은 기간(어제~과거 days일) 동안 MA5 < MA20 이 '항상' 성립
    컬럼 필요: '등락률', 'MA5', 'MA20'
    """

    # 안전 변환
    chg  = pd.to_numeric(data['등락률'], errors='coerce')
    ma5  = pd.to_numeric(data['MA5'],   errors='coerce')
    ma20 = pd.to_numeric(data['MA20'],  errors='coerce')

    # 최소 길이: 오늘 1 + 과거 days
    if len(data) < days + 1:
        return False

    # 오늘 등락률(마지막 행)
    today_chg = chg.iloc[-1]

    # 어제~과거 days일 (총 days개): 마지막 행 제외한 꼬리 days개
    past_chg  = chg.iloc[-(days+1):-1]
    past_ma5  = ma5.iloc[-(days+1):-1]
    past_ma20 = ma20.iloc[-(days+1):-1]

    # 결측 있으면 보수적으로 False (원하면 dropna로 완화 가능)
    if past_chg.isna().any() or past_ma5.isna().any() or past_ma20.isna().any() or pd.isna(today_chg):
        return False

    cond_today        = (today_chg >= up_thr)
    cond_past_anydrop = past_chg.le(down_thr).any()     # 하루라도 -4% 이하
    cond_ma_order     = past_ma5.lt(past_ma20).all()    # 12일 내내 MA5 < MA20

    return bool(cond_today and cond_past_anydrop and cond_ma_order)



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

today = datetime.today().strftime('%Y%m%d')

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['066970']
# tickers = ['114190']
tickers = ['044480']

# 결과를 저장할 배열
results = []


for count, ticker in enumerate(tickers):
    condition_passed = True
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)

    data = df

    # 디버깅용
    # data = data[:-9]
    # print(data[:-1])

    ########################################################################

    closes = data['종가'].values
    trading_value = data['거래량'] * data['종가']


    # 데이터가 부족하면 패스
    if data.empty or len(data) < 50:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    data = cleaned

    data, removed_idx = drop_trading_halt_rows(data)


    if 'MA5' not in data.columns or 'MA20' not in data.columns:
        continue

    # 5일선은 20일선보다 낮아야 한다
    ma5_today = data['MA5'].iloc[-1]
    ma20_today = data['MA20'].iloc[-1]

    if ma5_today >= ma20_today:
        continue

    signal = signal_any_drop(data)
    if not signal:
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

    plot_candles_weekly(data, show_months=12, title=f'{today} {stock_name} [{ticker}] Weekly Chart',
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

