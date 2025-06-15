from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import fetch_stock_data


'''
📌 “과매도, 매수 신호”와 이동평균선의 방향
1. 이론상
  과매도/매수 신호(예: 볼린저밴드 하단 이탈, RSI 30 미만 등)는 “주가가 최근 평균에 비해 너무 많이 빠진 상태”를 의미합니다.
  이때 이동평균선(MA)은 “상승/하락 중일 수도, 횡보 중일 수도” 있습니다.

2. 실전에서의 “베스트 신호”
  이동평균선이 하락(우하향) 중일 때 과매도 신호가 뜨면:
    실제로는 “하락 추세 속의 과매도”
    반등 신호로 잘 안 통하고, 하락 추세가 더 이어질 수 있음

  이동평균선이 횡보 혹은 “상승 전환 직후”에 과매도 신호가 뜨면:
    반등 확률이 훨씬 높음!
    즉, “추세가 꺾이거나 바닥을 다진 다음” 과매도 신호가 진짜 먹히는 경우가 많음

3. 매매 실전에서는?
  이동평균선(특히 20일선 등)이 하락 중이면 과매도 신호는 “추세 매수”보다는 단기 반등(데드캣바운스)만 노리는 데 적합
  진짜 좋은 매수 신호는
    MA가 횡보/상승 전환 + 과매도(밴드 하단, RSI 과매도 등)
    즉, 하락이 멈췄거나 이미 바닥 찍은 후!

✅ 요약
과매도 신호만으로 매수하는 건 “추세 무시 단기 반등 노림”에 불과함
MA가 횡보/상승 + 과매도 신호 = 가장 신뢰도 높은 매수 신호
MA가 하락 중일 때 과매도 신호는 “하락 추세 속 과매도”이므로 신중해야 함

🚩 실전 팁
과매도+MA 하락= 단기반등(조심), MA 상승+과매도= 강력 반등 신호
두 조건을 AND로 쓰는 게 실전에서 승률이 더 높음

“조건 조합 매매전략 코드”,
“실제 주가 차트 예시”,
“과매도+MA방향 필터 실전 로직”
'''


window_10 = 10
num_std = 2

# 데이터 수집
# today = datetime.today().strftime('%Y%m%d')
today = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=100)).strftime('%Y%m%d')
# ticker = "002380"
ticker = "000660"

data = fetch_stock_data(ticker, last_year, today)
# data.to_pickle(f'{ticker}.pkl')
# data = pd.read_pickle(f'{ticker}.pkl')


# 볼린저밴드 10
data['MA10'] = data['종가'].rolling(window=window_10).mean()
data['STD10'] = data['종가'].rolling(window=window_10).std()
data['UpperBand10'] = data['MA10'] + (num_std * data['STD10'])
data['LowerBand10'] = data['MA10'] - (num_std * data['STD10'])

# 현재가
last_close = data['종가'].iloc[-1]
upper10 = data['UpperBand10'].iloc[-1]
lower10 = data['LowerBand10'].iloc[-1]

# 매수/매도 조건 (둘 다 확인)
if last_close <= lower10:
    print("과매도, 매수 신호!")
elif last_close >= upper10:
    print("과매수, 매도 신호!")
else:
    print("중립(관망)")

MA10_color = 'gray'
plt.figure(figsize=(12,6))
plt.plot(data['종가'], label='Close Price')
plt.plot(data['MA10'], label='MA10')
plt.plot(data['UpperBand10'], label='UpperBand10 (2σ)', linestyle='--', color=MA10_color)
plt.plot(data['LowerBand10'], label='LowerBand10 (2σ)', linestyle='--', color=MA10_color)
plt.fill_between(data.index, data['UpperBand10'], data['LowerBand10'], color=MA10_color, alpha=0.2)
plt.legend()
plt.title('Bollinger Bands 10')
plt.show()