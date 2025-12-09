from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os, sys
from pathlib import Path

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
    else:
        raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data

window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

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
data = data.dropna(subset=['종가', '거래량'])


data['MA20'] = data['종가'].rolling(window=window).mean()
data['STD20'] = data['종가'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])
data['MA5'] = data['종가'].rolling(window=5).mean()
data['MA10'] = data['종가'].rolling(window=10).mean()
data['MA15'] = data['종가'].rolling(window=15).mean()


# 조건별 색상 결정
up = data['종가'] > data['시가']
down = data['종가'] < data['시가']

bar_colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))  # 같은 값은 gray

data_plot = data.copy()
data_plot['date_str'] = data_plot.index.strftime('%Y-%m-%d') # 각 행의 인덱스를 문자열로 변환해서 'date_str'칼럼에 추가


# 그래프 (윗부분: 가격, 아랫부분: 거래량)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- 상단: 가격 + 볼린저밴드 ---
ax1.plot(data_plot['date_str'], data_plot['종가'], label='Close Price', marker='s', markersize=6, markeredgecolor='white')
ax1.plot(data_plot['date_str'], data_plot['MA20'], label='MA20', alpha=0.8)
ax1.plot(data_plot['date_str'], data_plot['MA5'], label='MA5', alpha=0.8)
ax1.plot(data_plot['date_str'], data_plot['UpperBand'], label='Upper Band (2σ)', linestyle='--', alpha=0.8)
ax1.plot(data_plot['date_str'], data_plot['LowerBand'], label='Lower Band (2σ)', linestyle='--', alpha=0.8)
ax1.fill_between(data_plot['date_str'], data_plot['UpperBand'], data_plot['LowerBand'], color='gray', alpha=0.18)

ax1.legend()
ax1.grid(True)
ax1.set_title('Bollinger Bands And Volume')

# --- 하단: 거래량 (양/음/동색 구분) ---
ax2.bar(data_plot['date_str'], data_plot['거래량'], color=bar_colors, alpha=0.7)
ax2.grid(True)

tick_idx = np.arange(0, len(data_plot), 10) # x days
ax2.set_xticks(data_plot['date_str'].iloc[tick_idx])

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

'''
fig: 전체 그림(figure) 객체
plt.subplots(2, 1): 2행 1열
ax1, ax2: 각각 위쪽, 아래쪽의 개별 그래프(axes) 객체
figsize: 인치
sharex=True: 두 그래프의 x축(가로축) 범위가 자동으로 동기화
'''