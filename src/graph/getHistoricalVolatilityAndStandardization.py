import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ccxt를 사용하여 Binance에서 데이터 가져오기
exchange = ccxt.binance()

def fetch_historical_prices(symbol, days=365):
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['close']

btc_close = fetch_historical_prices('BTC/USDT')

# 일별 수익률 계산
btc_returns = btc_close.pct_change()

# 30일간의 역사적 변동성 계산
volatility_30d = btc_returns.rolling(window=30).std() * np.sqrt(365)

# 데이터 표준화
scaler = StandardScaler()
btc_close_scaled = scaler.fit_transform(btc_close.values.reshape(-1, 1))
volatility_30d_scaled = scaler.fit_transform(volatility_30d.values.reshape(-1, 1))

# 시각화
plt.figure(figsize=(14, 7))

plt.plot(btc_close.index, btc_close_scaled, label='BTC Close Price (Standardized)', linewidth=2)
plt.plot(volatility_30d.index, volatility_30d_scaled, label='BTC 30-Day Volatility (Standardized)', linewidth=2, linestyle='--')

plt.title('Standardized Bitcoin Close Price vs. 30-Day Historical Volatility')
plt.legend()
plt.show()
