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

# ticker = "002380"
ticker = "000660"

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