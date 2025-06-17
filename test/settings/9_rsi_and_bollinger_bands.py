import yfinance as yf
from datetime import datetime, timedelta
import pytz
import sys

window=20
num_std=2
ticker = 'RKLB'
# 입력 25, 예측 5

now_us = datetime.now(pytz.timezone('America/New_York'))
start_date_us = (now_us - timedelta(days=200)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

df = yf.download(ticker, start=start_date_us, end=end_date)
print('dataset', len(df))

# 데이터가 비어 있으면 안내
if df.empty:
    print("데이터가 없습니다! (티커 혹은 날짜, 네트워크 문제일 수 있습니다)")
    sys.exit(0)
else:
    # 필요시 주요 컬럼만 뽑아서 사용
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]


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

# 볼린저 밴드
data['MA20'] = data['Close'].rolling(window=window).mean()
data['STD20'] = data['Close'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

# 현재가
last_close = data['Close'].iloc[-1]
upper = data['UpperBand'].iloc[-1]
lower = data['LowerBand'].iloc[-1]

plt.figure(figsize=(14,8))

# 가격 차트 (상단)
plt.subplot(2,1,1)
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA20'], label='MA20')
plt.plot(data['UpperBand'], label='Upper Band (2σ)', linestyle='--')
plt.plot(data['LowerBand'], label='Lower Band (2σ)', linestyle='--')
plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)
plt.legend()
plt.grid(True)
plt.title('Bollinger Bands')


# RSI 차트 (하단)
plt.subplot(2,1,2)
plt.plot(data['RSI'], label='RSI(14)', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='blue', linestyle='--', label='Oversold (30)')
plt.title('RSI (Relative Strength Index)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()