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
import pickle

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 예측 결과를 실제 값(주가)으로 복원
def invert_scale(scaled_preds, scaler, feature_index=3):
    """
    scaled_preds: (샘플수, forecast_horizon) - 스케일된 종가 예측 결과
    scaler: 학습에 사용된 MinMaxScaler 객체
    feature_index: 종가 컬럼 인덱스(보통 3)
    """
    inv_preds = []
    for row in scaled_preds:
        temp = np.zeros((len(row), scaler.n_features_in_))
        temp[:, feature_index] = row  # 종가 위치에 예측값 할당
        inv = scaler.inverse_transform(temp)[:, feature_index]  # 역변환 후 종가만 추출
        inv_preds.append(inv)
    return np.array(inv_preds)


# 데이터 수집
ticker = '000660' # 하이닉스
data = fdr.DataReader(ticker, '2025-01-01', '2025-06-03')

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
sequence_length = 20  # 거래일 기준 약 60일 (3개월)
forecast_horizon = 1  # 앞으로 5일 예측 (영업일 기준 일주일)

# 시계열 데이터에서, 최근 N일 데이터를 가지고 미래 M일치를 예측하는 딥러닝 구조에 맞는 방식으로 변환
X = [] # 과거 60일의 모든 특성
y = [] # 미래 7일의 예측 종가
# for i in range(len(scaled_data) - sequence_length):
#     X.append(scaled_data[i:i+sequence_length])
#     y.append(scaled_data[i+sequence_length, 3])  # 종가(Close) 예측
for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
    X.append(scaled_data[i:i+sequence_length])
    # 7일치 종가 [D+1 ~ D+7] > Dense(7)로 만들려면 y값도 (샘플수, 7) 로 만들어야 한다
    y.append(scaled_data[i+sequence_length:i+sequence_length+forecast_horizon, 3]) # 종가 예측

X = np.array(X)
Y = np.array(y)

# 학습 데이터와 검증 데이터 분리
# random_state; 데이터를 랜덤하게 분할할 때의 랜덤 시드(seed) 값 > 항상 같은 방식으로 데이터를 나눔 (재현성 보장)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(forecast_horizon))

model.compile(optimizer='adam', loss='mean_squared_error')


# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min') # 학습(fit()) 도중 검증 손실(val_loss)이 가장 낮은 시점의 모델을 자동 저장

history2 = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=16,
    verbose=0,
    validation_data=(X_val, Y_val),
    shuffle=False,
#     callbacks=[early_stop, checkpoint]
    callbacks=[early_stop]
)

# 실제 값과 예측 값의 비교, 7 일치 평균 비교, 실제 값으로 복원
predictions = model.predict(X_val)
predictions_inv = invert_scale(predictions, scaler)
Y_val_inv = invert_scale(Y_val, scaler)
plt.figure(figsize=(10, 5))
plt.title(f'{ticker} Stock Price Prediction')
plt.plot(Y_val_inv.mean(axis=1), label='Actual Mean')
plt.plot(predictions_inv.mean(axis=1), label='Predicted Mean')
plt.xlabel(f'Next {forecast_horizon} days')
plt.ylabel('Price')
plt.legend()
plt.show()