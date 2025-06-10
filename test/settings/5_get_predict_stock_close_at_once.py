import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
import os

today = datetime.today().strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=60)).strftime('%Y%m%d')
ticker = "214450"

ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental['PER'] = fundamental['PER'].fillna(0)
data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_multistep_dataset(dataset, look_back=25, n_future=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back:i+look_back+n_future, 3])
    return np.array(X), np.array(Y)

n_future = 5
look_back = 25
X, Y = create_multistep_dataset(scaled_data, look_back=look_back, n_future=n_future)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(n_future)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model_path = 'save_weights.h5'

if os.path.exists(model_path):
    model.load_weights(model_path)
    print(f"{ticker}: 이전 가중치 로드")

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, Y, batch_size=16, epochs=200, validation_split=0.2, verbose=0, callbacks=[early_stop]) # fit; 학습 > 저장

model.save_weights(model_path)
print(f"{ticker}: 학습 후 가중치 저장됨 -> {model_path}")

# 종가 scaler fit (실제 데이터로)
close_scaler = MinMaxScaler()
close_prices = data['종가'].values.reshape(-1, 1)
close_scaler.fit(close_prices)

# 미래 5일 예측
X_input = scaled_data[-look_back:].reshape(1, look_back, scaled_data.shape[1])
future_preds = model.predict(X_input, verbose=0).flatten()
future_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten()

print("미래 5일 예측 종가:", future_prices)

# 날짜 처리
future_dates = pd.date_range(start=ohlcv.index[-1] + pd.Timedelta(days=1), periods=n_future, freq='B') # 예측할 5 영업일 날짜
actual_prices = ohlcv['종가'].values # 최근 종가 배열

plt.figure(figsize=(10, 5))
plt.plot(ohlcv.index, actual_prices, label='Actual Prices') # 과거; ohlcv.index 받아온 날짜 인덱스
plt.plot(future_dates, future_prices, label='Future Predicted Prices', linestyle='--', marker='o', color='orange') # 예측

# 마지막 실제값과 첫 번째 예측값을 점선으로 연결
plt.plot(
    [ohlcv.index[-1], future_dates[0]],  # x축: 마지막 실제날짜와 첫 예측날짜
    [actual_prices[-1], future_prices[0]],  # y축: 마지막 실제종가와 첫 예측종가
    linestyle='dashed', color='gray', linewidth=1.5
)

plt.title(f'Actual and 5-day Predicted Prices for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
# plt.show()




output_dir = 'D:\\stocks'
last_price = data['종가'].iloc[-1]
future_return = (future_prices[-1] / last_price - 1) * 100
stock_name = stock.get_market_ticker_name(ticker)
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
file_path = os.path.join(output_dir, f'5 {today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
plt.savefig(file_path)
plt.close()