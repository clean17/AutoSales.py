import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
ticker = "012750"  # 에스원 선택

# 주요 피처(시가, 고가, 저가, 종가, 거래량) + 재무 지표(PER)
ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)

# PER 값이 없는 경우 대체 값 사용 (예: 0으로 채우기)
fundamental['PER'] = fundamental['PER'].fillna(0)

# 주가 데이터와 재무 지표 결합 및 NaN 값 처리
data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
print('######', len(data))

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print('######', len(scaled_data))

# 시계열 데이터를 윈도우로 나누기
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset(scaled_data, look_back)

# 모델 생성 및 학습
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dense(128),
    Dense(64),
    Dense(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, batch_size=32, epochs=50, validation_split=0.1)

# 모델 저장
model_path = 'my_model1.keras'
model.save(model_path)

# 모델 로드 및 예측
model_loaded = load_model(model_path)
predictions = model_loaded.predict(X[-7:]).flatten()

# 종가 데이터에 대한 스케일러 생성 및 훈련
close_scaler = MinMaxScaler()
close_prices = data['종가'].values.reshape(-1, 1)
close_scaler.fit(close_prices)

# 종가 예측값만 스케일 역변환
predictions_scaled = predictions.reshape(-1, 1)
predictions = close_scaler.inverse_transform(predictions_scaled).flatten()

# 실제 값과 예측 값 비교를 위한 그래프
actual_prices = ohlcv['종가'].values

# 예측 부분을 실제 데이터에 자연스럽게 연결
connected_predictions = np.insert(predictions, 0, actual_prices[-1])

# 날짜 처리
last_date = ohlcv.index[-1]
prediction_dates = pd.date_range(start=last_date, periods=len(predictions) + 1)  # 예측 시작 날짜 포함

# 시각화
plt.figure(figsize=(16, 8))
plt.plot(ohlcv.index, actual_prices, label='Actual Prices')
plt.plot(prediction_dates, connected_predictions, label='Predicted Prices', linestyle='--')
plt.title(f'Actual and Predicted Prices for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()