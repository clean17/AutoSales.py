import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ccxt 라이브러리를 사용하여 Binance 거래소 객체 생성
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# 데이터를 가져올 함수 정의
def fetch_closing_prices(symbol, days=365):
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
    df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df['close']

# 비트코인의 종가 데이터 가져오기
btc_close = fetch_closing_prices('BTC/USDT')

# 일별 수익률 계산
daily_returns = btc_close.pct_change()

# 30일간의 역사적 변동성 계산
volatility = daily_returns.rolling(window=30).std() * np.sqrt(365)

# 변동성 시각화
plt.figure(figsize=(14, 7))
volatility.plot()
plt.title('Bitcoin 30-Day Historical Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()

'''
최근 (24.04.06) 역사적 변동성은 0.7 => 대략 70% 변동성
비트코인 영업일 365의 제곱근은 대략 19
70 / 19 = 대략 3.7%
=> 하루 평균 3.7%가 변동되었다.
'''
