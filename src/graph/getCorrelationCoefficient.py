'''
비트코인과 이더리움의 상관관계 그래프 보기
'''
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ccxt를 사용하여 거래소 객체 생성
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'timeout': 30000,  # 30초
})

# 데이터를 가져올 함수 정의
def fetch_closing_prices(symbol, days=365):
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since)
    df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df['close']

# 비트코인과 이더리움의 종가 데이터 가져오기
btc = fetch_closing_prices('BTC/USDT')
eth = fetch_closing_prices('ETH/USDT')

# 데이터 프레임 생성
data = pd.DataFrame({'BTC': btc, 'ETH': eth})

# 상관계수 계산
correlation = data.corr().iloc[0, 1]
print(f"비트코인과 이더리움의 상관계수: {correlation:.2f}")

# 데이터 표준화 StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=['BTC', 'ETH'], index=data.index)

# 시각화 matplotlib
plt.figure(figsize=(14, 7))
plt.plot(data_scaled['BTC'], label='Bitcoin (Standardized)', linewidth=2)
plt.plot(data_scaled['ETH'], label='Ethereum (Standardized)', linewidth=2)
plt.title('Standardized Prices of Bitcoin and Ethereum Over the Last Year')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Standardized Price')
plt.grid(True)
plt.show()
