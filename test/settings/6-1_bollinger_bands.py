from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

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
# today = datetime.today().strftime('%Y%m%d')
today = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=100)).strftime('%Y%m%d')
ticker = "000660"


data = fetch_stock_data(ticker, last_year, today)
# data.to_pickle(f'{ticker}.pkl')
# data = pd.read_pickle(f'{ticker}.pkl')

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


plt.figure(figsize=(12,6))
plt.plot(data['종가'], label='Close Price')
plt.plot(data['MA20'], label='MA20')
plt.plot(data['UpperBand'], label='Upper Band (2σ)', linestyle='--')
plt.plot(data['LowerBand'], label='Lower Band (2σ)', linestyle='--')
plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)
plt.legend()
plt.title('Bollinger Bands')
plt.show()