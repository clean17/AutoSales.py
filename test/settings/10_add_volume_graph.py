from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import fetch_stock_data

window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
# today = (datetime.today() - timedelta(days=3)).strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=100)).strftime('%Y%m%d')
ticker = "007860"


data = fetch_stock_data(ticker, last_year, today)
data.to_pickle(f'{ticker}.pkl')
data = pd.read_pickle(f'{ticker}.pkl')
data = data.dropna(subset=['종가', '거래량'])

print(data)

data['MA20'] = data['종가'].rolling(window=window).mean()
data['STD20'] = data['종가'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

data['MA5'] = data['종가'].rolling(window=5).mean()
data['MA10'] = data['종가'].rolling(window=10).mean()
data['MA15'] = data['종가'].rolling(window=15).mean()

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
#
# # --- 상단: 가격 & 볼린저밴드 ---
# ax1.plot(data['종가'], label='Close Price')
# ax1.plot(data['MA20'], label='MA20')
# ax1.plot(data['MA15'], label='MA15')
# # ax1.plot(data['MA10'], label='MA10')
# ax1.plot(data['MA5'], label='MA5')
# ax1.plot(data['UpperBand'], label='Upper Band (2σ)', linestyle='--')
# ax1.plot(data['LowerBand'], label='Lower Band (2σ)', linestyle='--')
# ax1.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)
# ax1.legend()
# ax1.set_title('Bollinger Bands')
#
# # --- 하단: 거래량 ---
# ax2.bar(data.index, data['거래량'], color='black', alpha=0.4)
# ax2.set_ylabel('Volume')
#
# plt.tight_layout()
# plt.show()

# 조건별 색상 결정
up = data['종가'] > data['시가']
down = data['종가'] < data['시가']

bar_colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))  # 같은 값은 gray

data_plot = data.copy()
data_plot['date_str'] = data_plot.index.strftime('%Y-%m-%d')

# 그래프 (윗부분: 가격, 아랫부분: 거래량)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1]})

# --- 상단: 가격 + 볼린저밴드 ---
# ax1.plot(data['종가'], label='Close Price')
ax1.plot(data_plot['date_str'], data_plot['종가'], label='Close Price')
ax1.plot(data_plot['date_str'], data_plot['MA20'], label='MA20')
ax1.plot(data_plot['date_str'], data_plot['MA5'], label='MA5')
ax1.plot(data_plot['date_str'], data_plot['UpperBand'], label='Upper Band (2σ)', linestyle='--')
ax1.plot(data_plot['date_str'], data_plot['LowerBand'], label='Lower Band (2σ)', linestyle='--')
ax1.fill_between(data_plot['date_str'], data_plot['UpperBand'], data_plot['LowerBand'], color='gray', alpha=0.2)

ax1.legend()
ax1.grid(True)
ax1.set_title('Bollinger Bands')

# --- 하단: 거래량 (양/음/동색 구분) ---
# ax2.bar(data.index, data['거래량'], color=bar_colors, alpha=0.7)
ax2.bar(data_plot['date_str'], data_plot['거래량'], color=bar_colors, alpha=0.7)
ax2.set_ylabel('Volume')
ax2.grid(True)

# data_plot['date_str']가 x축 전체 라벨입니다.
tick_idx = np.arange(0, len(data_plot), 5) # 5 days
ax2.set_xticks(data_plot['date_str'].iloc[tick_idx])

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()