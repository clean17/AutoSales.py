from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
# ticker = "005930"  # 삼성전자
ticker = "012750"  # 에스원

ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)
data = pd.concat([ohlcv['종가'], fundamental['PER']], axis=1).dropna()

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(scaled_data, look_back)

# LSTM 모델 생성 및 학습
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, batch_size=32, epochs=20) # epochs 학습 횟수 batch_size 학습에 사용할 샘플의 수

# 주가 예측
predictions = model.predict(X[-7:])

# 예측 결과 시각화
predicted_prices = scaler.inverse_transform(np.concatenate((predictions, np.zeros((7, 1))), axis=1))[:, 0]
actual_prices = data['종가'].values[-7:]

# 마지막 실제 가격 데이터 날짜 가져오기
last_date = ohlcv.index[-1]

# 예측된 가격을 위한 날짜 생성
prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=8)

# 예측 결과 시각화 수정
plt.figure(figsize=(20, 6))

# 실제 가격 데이터 플롯
plt.plot(ohlcv.index, ohlcv['종가'], label='Actual Prices', color='blue')

# 예측 시작 지점에 실제 데이터의 마지막 가격을 포함하여 연속적으로 만들기
predicted_prices_with_continuity = np.insert(predicted_prices, 0, ohlcv['종가'].iloc[-1])

# 수정된 예측 가격 데이터 플롯
plt.plot(prediction_dates, predicted_prices_with_continuity, label='Predicted Prices', color='red', linestyle='--')

plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()