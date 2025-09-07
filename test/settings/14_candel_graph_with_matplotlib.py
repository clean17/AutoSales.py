from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
from pathlib import Path
from matplotlib.patches import Rectangle


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, add_technical_features



# 데이터 수집
ticker = "007860"


DATA_COLLECTION_PERIOD = 400 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
today = datetime.today().strftime('%Y%m%d')

# 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
if os.path.exists(filepath):
    df = pd.read_pickle(filepath)
    data = fetch_stock_data(ticker, start_five_date, today)
else:
    df = pd.DataFrame()
    data = fetch_stock_data(ticker, start_date, today)

# dropna; pandas DataFrame에서 결측값(NaN)이 있는 행을 모두 삭제; 받아오는 데이터가 영업일 기준이므로 할 필요가 없다
data = add_technical_features(data)
# 하나라도 결측이 있으면 행을 삭제
data = data.dropna(subset=['종가', '거래량'])



# df = data.copy()
#
# # 최근 5개월만
# end   = df.index.max()
# start = end - pd.DateOffset(months=5)
# df = df.loc[start:end].copy()
# df.index = pd.to_datetime(df.index)
#
# # ==== 색상 기준: 종가 vs 시가 ====
# openp  = df['시가'].to_numpy()
# closep = df['종가'].to_numpy()
# high   = df['고가'].to_numpy()
# low    = df['저가'].to_numpy()
#
# up   = closep > openp
# down = closep < openp
# same = ~(up | down)
#
# colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))
#
# # x축 위치
# x = np.arange(len(df))
# date_str = df.index.strftime('%Y-%m-%d')
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12),
#                                sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#
# # --- 윗부분: 캔들 + 지표 ---
#
# # 1) 윅(저~고) - 방향별 색상 적용
# ax1.vlines(x[up],   low[up],   high[up],   color='red',  linewidth=1, alpha=0.9)
# ax1.vlines(x[down], low[down], high[down], color='blue', linewidth=1, alpha=0.9)
# ax1.vlines(x[same], low[same], high[same], color='gray', linewidth=1, alpha=0.9)
#
# # 2) 몸통(시가↔종가) - 방향별 색상 적용
# width = 0.6
# for i, (o, c, col) in enumerate(zip(openp, closep, colors)):
#     bottom = min(o, c)
#     height = max(abs(c - o), 1e-8)  # 0폭 방지
#     ax1.add_patch(Rectangle((x[i] - width/2, bottom), width, height,
#                             facecolor=col, edgecolor=col, alpha=0.9))
#
# # (옵션) 이동평균/볼밴 있으면 오버레이
# if 'MA20' in df.columns:
#     ax1.plot(x, df['MA20'], label='MA20', alpha=0.9)
# if 'MA5' in df.columns:
#     ax1.plot(x, df['MA5'], label='MA5', alpha=0.9)
# if {'UpperBand','LowerBand'}.issubset(df.columns):
#     ax1.plot(x, df['UpperBand'], label='Upper Band (2σ)', linestyle='--', alpha=0.8)
#     ax1.plot(x, df['LowerBand'], label='Lower Band (2σ)', linestyle='--', alpha=0.8)
#     ax1.fill_between(x, df['UpperBand'], df['LowerBand'], color='gray', alpha=0.18)
#
# ax1.set_title('Bollinger Bands And Volume (Candlestick — close vs open)')
# ax1.grid(True)
# ax1.legend()
#
# # --- 아랫부분: 거래량 (윗부분과 동일 색) ---
# ax2.bar(x, df['거래량'], color=colors, alpha=0.7)
# ax2.set_ylabel('Volume')
# ax2.grid(True)
#
# # x축 라벨(10개 간격)
# tick_idx = np.arange(0, len(df), 10)
# ax2.set_xticks(tick_idx)
# ax2.set_xticklabels(date_str[tick_idx], rotation=45, ha='right')
#
# plt.tight_layout()
# plt.show()



def plot_candles_standard(data: pd.DataFrame, months=5, title="Candlestick (KRX style)"):
    df = data.copy()

    # 0) 전처리: 인덱스/정렬/숫자화
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    for col in ['시가','고가','저가','종가','거래량']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['시가','고가','저가','종가'])  # 필수값 없는 행 제거

    # 1) 최근 N개월만
    end = df.index.max()
    start = end - pd.DateOffset(months=months)
    df = df.loc[start:end].copy()
    print(df)

    # 2) 색상(상승=빨강, 하락=파랑, 보합=회색) — 기준: 종가 vs 시가
    openp  = df['시가'].to_numpy()
    closep = df['종가'].to_numpy()
    high   = df['고가'].to_numpy()
    low    = df['저가'].to_numpy()

    up   = closep > openp
    down = closep < openp
    same = ~(up | down)
    body_colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))
    wick_colors = body_colors  # 국내 증권사와 동일하게 윅도 몸통색과 맞춤

    # 3) x축은 등간격(거래일만 존재하므로 문제 없음)
    x = np.arange(len(df))
    date_str = df.index.strftime('%Y-%m-%d')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12),
                                   sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 윗부분: 표준 캔들 ---
    width = 0.6  # 몸통 폭

    # (A) 윅을 "몸통 밖" 구간만 2분할로 그려, 몸통과 겹쳐도 깔끔하게 보이도록
    for i in range(len(df)):
        c = wick_colors[i]
        o, h, l, c_ = openp[i], high[i], low[i], closep[i]
        top = max(o, c_)
        bot = min(o, c_)
        # 윗윅: top ~ high
        if h > top:
            ax1.vlines(x[i], top, h, color=c, linewidth=1.0, alpha=0.95, zorder=1)
        # 아랫윅: low ~ bot
        if l < bot:
            ax1.vlines(x[i], l, bot, color=c, linewidth=1.0, alpha=0.95, zorder=1)

    # (B) 몸통: 시가↔종가만 사각형
    for i in range(len(df)):
        o, c_ = openp[i], closep[i]
        bottom = min(o, c_)
        height = max(abs(c_ - o), 1e-8)   # 0폭 방지
        ax1.add_patch(
            Rectangle(
                (x[i] - width/2, bottom), width, height,
                facecolor=body_colors[i], edgecolor=body_colors[i],
                linewidth=1.0, alpha=0.95, zorder=2  # 몸통이 윅 위로
            )
        )

    # (옵션) 이동평균/볼밴 오버레이 — 인덱스 맞춰서 그리기
    if 'MA20' in df.columns:
        ax1.plot(x, df['MA20'].to_numpy(), label='MA20', alpha=0.9, zorder=0)
    if 'MA5' in df.columns:
        ax1.plot(x, df['MA5'].to_numpy(),  label='MA5',  alpha=0.9, zorder=0)
    if {'UpperBand','LowerBand'}.issubset(df.columns):
        ub = df['UpperBand'].to_numpy()
        lb = df['LowerBand'].to_numpy()
        ax1.plot(x, ub, '--', color='gray', alpha=0.85, label='UpperBand', zorder=0)
        ax1.plot(x, lb, '--', color='gray', alpha=0.85, label='LowerBand', zorder=0)
        ax1.fill_between(x, ub, lb, color='gray', alpha=0.18, zorder=-1)

    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper left')

    # --- 아랫부분: 거래량 (몸통 색과 동일)
    ax2.bar(x, df['거래량'].to_numpy(), color=body_colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.25)

    # x축 라벨(가독성)
    tick_idx = np.arange(0, len(df), max(1, len(df)//12))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(date_str[tick_idx], rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


plot_candles_standard(data, months=5, title='Bollinger Bands & Volume — Standard Candles')