import yfinance as yf

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

from utils import compute_rsi

ticker = 'RKLB'
start_date = '2024-01-01'
end_date = '2024-06-14'

df = yf.download(ticker, start=start_date, end=end_date)

# 데이터 확인
# print(df.head())   # 상위 5개 데이터 출력
# print(df.tail())   # 하위 5개 데이터 출력

# 데이터가 비어 있으면 안내
if df.empty:
    print("데이터가 없습니다! (티커 혹은 날짜, 네트워크 문제일 수 있습니다)")
else:
    # 필요시 주요 컬럼만 뽑아서 사용
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # print(df)


import matplotlib.pyplot as plt


# 예시: 종가 데이터프레임
data = df

# data = fetch_stock_data(ticker, last_year, today)
data['RSI'] = compute_rsi(data['Close'])

plt.figure(figsize=(14,8))

# 가격 차트 (상단)
plt.subplot(2,1,1)
plt.plot(data['Close'], label='Close Price')
plt.title('Close Price')
plt.legend()

# RSI 차트 (하단)
plt.subplot(2,1,2)
plt.plot(data['RSI'], label='RSI(14)', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='blue', linestyle='--', label='Oversold (30)')
plt.title('RSI (Relative Strength Index)')
plt.legend()
plt.tight_layout()
plt.show()