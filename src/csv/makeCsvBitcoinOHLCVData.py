import ccxt
import pandas as pd
from datetime import datetime, timedelta

# ccxt 라이브러리를 사용하여 Binance 거래소 객체 생성
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# 시간 설정
끝_날짜 = datetime.utcnow()
시작_날짜 = 끝_날짜 - timedelta(days=365)

# Binance에서 OHLCV 데이터 가져오기
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=exchange.parse8601(시작_날짜.strftime('%Y-%m-%d %H:%M:%S')))

# 데이터 프레임으로 변환
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# CSV 파일로 저장
df.to_csv('BTC_OHLCV_1년간_데이터.csv', index=False)

print("CSV 파일이 저장되었습니다.")