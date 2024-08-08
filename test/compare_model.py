import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 데이터 수집
ticker = '000150'  # 예시: 000150 종목 코드
data = fdr.DataReader(ticker, '2020-01-01', '2023-09-20')

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
sequence_length = 60  # 예: 최근 60일 데이터를 기반으로 예측
X = []
y = []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length, 3])  # 종가(Close) 예측

X = np.array(X)
Y = np.array(y)

# 학습 데이터와 검증 데이터 분리
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
# model = Sequential([
#     LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     LSTM(128, return_sequences=True),
#     LSTM(64, return_sequences=False),
#     Dense(128),
#     Dense(64),
#     Dense(32),
#     Dense(1)
# ])
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),  # 드롭아웃 추가
    LSTM(128, return_sequences=True),
    Dropout(0.2),  # 드롭아웃 추가
    LSTM(64, return_sequences=False),
    Dense(128),
    Dropout(0.2),  # 드롭아웃 추가
    Dense(64),
    Dense(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model_file_path = 'my_model.h5'
checkpoint = ModelCheckpoint(
    model_file_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1,
          validation_data=(X_val, Y_val), callbacks=[checkpoint])

# 모델 저장
# model.save('my_model.h5')
# 모델 로드
# model_loaded = load_model('my_model.h5') # 체크포인트가 자동으로 저장

# 새로운 데이터 예측
predictions = model.predict(X_val)

# 실제 값과 예측 값의 비교
plt.figure(figsize=(24, 10))
plt.plot(Y_val, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
