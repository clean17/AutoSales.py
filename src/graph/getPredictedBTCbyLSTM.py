import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# ccxt를 사용하여 거래소에서 비트코인 데이터 가져오기
'''
Binance API를 사용하여 지난 1년간의 비트코인 일별 OHLCV 데이터를 가져온다
'''
exchange = ccxt.binance()
symbol = 'BTC/USDT'

ohlcv_data = exchange.fetch_ohlcv(symbol, '1d', since=exchange.parse8601((datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')))
ohlcv_df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
ohlcv_df.set_index('timestamp', inplace=True)

# 데이터 전처리
'''
종가 데이터를 0과 1 사이의 값으로 스케일링
'''
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = ohlcv_df['close'].values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_prices)

# 시퀀스 데이터 준비
'''
시퀀스 데이터를 생성
look_back 매개변수는 예측을 위해 사용할 이전 데이터 포인트의 수
'''
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(scaled_close, look_back)

# LSTM 모델 생성 및 학습
'''
LSTM 모델을 생성하고 학습
두 개의 LSTM 층과 하나의 Dense 층으로 구성된 신경망
'''
X = X.reshape(X.shape[0], X.shape[1], 1)  # 샘플 수, 타임 스텝 수, 특성 수
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=100, batch_size=32)

# 다음 7일 예측
'''
학습된 모델을 사용하여 다음 7일간의 종가를 예측
'''
predictions = []
current_batch = X[-1].reshape(1, look_back, 1)

for i in range(7):
    current_pred = model.predict(current_batch)[0]
    predictions.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

# 예측 결과 스케일링 역변환
predictions = scaler.inverse_transform(predictions)

# 예측 결과 시각화를 위한 날짜 데이터 생성
last_date = ohlcv_df.index[-1]
prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)

# 실제 종가 및 예측 종가 플롯
'''
실제 종가 (파란색 선): 지난 1년간의 비트코인 실제 종가
예측된 종가 (빨간색 점선): LSTM 모델을 사용하여 예측한 다음 7일간의 비트코인 종가
'''

plt.figure(figsize=(24, 6))
plt.plot(ohlcv_df.index, ohlcv_df['close'], label='Actual Close Price', color='blue')
plt.plot(prediction_dates, predictions, label='Predicted Close Price', color='red', linestyle='--')

plt.title('Bitcoin Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()
