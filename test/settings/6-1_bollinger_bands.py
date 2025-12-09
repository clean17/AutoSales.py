from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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

'''
  [볼린저밴드(Bollinger Bands)]

주가가 어느 정도 '평균선'에서 벗어나 있는지
[과매수/과매도] 상태를 시각적으로 보여주는 대표적 보조지표

- 중심선: N일 이동평균선 (보통 20일)
- 상단밴드: 중심선 + (표준편차 × K) (보통 K=2)
- 하단밴드: 중심선 - (표준편차 × K)



매수/매도 신호 예시

주가가 하단 밴드에 근접/이탈:
>>> "과매도", 반등 기대 → 매수 신호로 보는 경우 많음

주가가 상단 밴드에 근접/이탈:
>>> "과매수", 하락 전환 기대 → 매도 신호로 보는 경우 많음
'''

window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
# today = (datetime.today() - timedelta(days=3)).strftime('%Y%m%d')
ticker = "007860"

DATA_COLLECTION_PERIOD = 400 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

# 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
if os.path.exists(filepath):
    df = pd.read_pickle(filepath)
    data = fetch_stock_data(ticker, start_five_date, today)
else:
    df = pd.DataFrame()
    data = fetch_stock_data(ticker, start_date, today)
print(data.tail(1))

data['MA20'] = data['종가'].rolling(window=window).mean()
data['STD20'] = data['종가'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

# 현재가
last_close = data['종가'].iloc[-1]
upper = data['UpperBand'].iloc[-1]
lower = data['LowerBand'].iloc[-1]

# 매수/매도 조건
if last_close <= lower:
    print("과매도, 매수 신호!")
elif last_close >= upper:
    print("과매수, 매도 신호!")
else:
    print("중립(관망)")


data['MA5'] = data['종가'].rolling(window=5).mean()
data['MA10'] = data['종가'].rolling(window=10).mean()
data['MA15'] = data['종가'].rolling(window=15).mean()

ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
ma_angle_10 = data['MA10'].iloc[-1] - data['MA10'].iloc[-2]
ma_angle_15 = data['MA15'].iloc[-1] - data['MA15'].iloc[-2]
ma_angle_20 = data['MA20'].iloc[-1] - data['MA20'].iloc[-2]

print(ma_angle_5)
# print(ma_angle_10)
print(ma_angle_15)
print(ma_angle_20)

plt.figure(figsize=(12,6))
plt.plot(data['종가'], label='Close Price')
plt.plot(data['MA20'], label='MA20')
plt.plot(data['MA15'], label='MA15')
# plt.plot(data['MA10'], label='MA10')
plt.plot(data['MA5'], label='MA5')
plt.plot(data['UpperBand'], label='Upper Band (2σ)', linestyle='--')
plt.plot(data['LowerBand'], label='Lower Band (2σ)', linestyle='--')
plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)
plt.legend()
plt.title('Bollinger Bands')
plt.show()