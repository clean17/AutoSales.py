import yfinance as yf

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


import pandas as pd
import matplotlib.pyplot as plt

def compute_rsi(prices, period=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

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