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

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 데이터 수집
# today = datetime.today().strftime('%Y%m%d')
today = (datetime.today() - timedelta(days=6)).strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=60)).strftime('%Y%m%d')
ticker = "000660"

# 주요 피처(시가, 고가, 저가, 종가, 거래량) + 재무 지표(PER)
ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)

# PER 값이 없는 경우 대체 값 사용 (예: 0으로 채우기)
fundamental['PER'] = fundamental['PER'].fillna(0)

# 주가 데이터와 재무 지표 결합 및 NaN 값 처리
data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_multistep_dataset(dataset, look_back, n_future):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back:i+look_back+n_future, 3])
    return np.array(X), np.array(Y)

n_future = 5
look_back = 25
X, Y = create_multistep_dataset(scaled_data, look_back, n_future)
# print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
# print("Y.shape:", Y.shape)
'''
X[0]: 0~9일 데이터 (과거 10일)
Y[0]: 10~14일 '종가' (미래 5일 정답)

X[1]: 1~10일 데이터
Y[1]: 11~15일 '종가'
'''

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


# 기존 모델이 있으면 불러와서 이어서 학습
model_path = 'models/save_model2.h5'
if os.path.exists(model_path):
    pass
#     model = tf.keras.models.load_model(model_path)
#     print(f"{ticker}: 이전 모델 로드")

# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원

# 이어서(혹은 처음부터) 학습
model.fit(X, Y, batch_size=16, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])

# 학습 후 모델 저장
# model.save(model_path)
# print(f"{ticker}: 학습 후 모델 저장됨 -> {model_path}")


# 종가 scaler fit (실제 데이터로)
close_scaler = MinMaxScaler()
close_prices = data['종가'].values.reshape(-1, 1)
close_scaler.fit(close_prices)

# 예측
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