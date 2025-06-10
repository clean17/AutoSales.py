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
data = fdr.DataReader(ticker, '2024-01-01', '2025-06-09')

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)

# 데이터 스케일링
'''
딥러닝, 머신러닝은 입력값의 크기에 민감 > 모델의 가중치 업데이트(학습)에 더 많은 영향을 끼치게 됨
모든 입력값이 비슷한 범위(주로 0~1)여야 신경망이 안정적으로 학습됨

특성마다 값의 범위를 맞춰서
신경망이 모든 입력을 골고루 잘 학습하게 하고,
학습 속도·성능·안정성을 크게 높이기 위해서
'''
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
sequence_length = 25  # 거래일 기준 약 60일 (3개월)
forecast_horizon = 5  # 앞으로 5일 예측 (영업일 기준 일주일)

# 시계열 데이터에서, 최근 N일 데이터를 가지고 미래 M일치를 예측하는 딥러닝 구조에 맞는 방식으로 변환
'''
|---60일---|---예측---|
[Day1~Day60] [Day61~Day67]
[Day2~Day61] [Day62~Day68]

이런 패턴이 오면, 이후 7일은 이렇게 된다! 를 통째로 배우게 됨
'''
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


'''
LSTM 레이어(노드) 수의 의미

많으면

모델이 더 많은 패턴(복잡한 시계열, 미세한 변화, 다양한 데이터 특성 등)을
기억하고 학습할 수 있음
과적합 위험↑
학습 속도↓, 메모리 사용↑

적으면

모델이 단순한/짧은/규칙적인 패턴을 빠르고 안정적으로 학습
과적합 위험↓
학습 속도↑, 자원 소모↓
'''
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))  # Dropout 비율도 늘려볼 것 추천
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

# model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.2))

# model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(forecast_horizon))

model.compile(optimizer='adam', loss='mean_squared_error')


# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min') # 학습(fit()) 도중 검증 손실(val_loss)이 가장 낮은 시점의 모델을 자동 저장

'''
epochs; 전체 데이터셋(X_train, Y_train)을 한 번 모두 학습하는 과정의 횟수 (많으면 과적합)
batch_size; 한 번에 모델에 넣어서 학습하는 데이터 샘플 수 (크면 효율, 불안정 작으면 일반화, 노이즈(과적합))
validation_data; 학습 도중 모델의 성능을 평가할 데이터셋

* Overfitting(과적합) 확인법
훈련 손실(loss)은 계속 떨어지는데
검증 손실(val_loss)은 어느 순간부터 더 이상 떨어지지 않고 오히려 올라갈 때

줄이는 방법
 Dropout 수치 증가 0.3 ~ 0.5
 LSTM 레이어(노드) 수 줄이기 64 > 32 > 16
 조기 종료(EarlyStopping) 콜백 사용
'''

old_history = None

# 체크포인트한 파일 존재하면 이어서 학습
model_file = 'best_model.h5'
history_file = 'history.pkl' # pkl; 파이썬 객체를 그대로 저장/불러오는 파일

if os.path.exists(model_file):
    model = load_model(model_file)
    print("기존 best_model.h5 불러옴, 이어서 학습/예측 가능")
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            old_history = pickle.load(f)
else:
    print(" *** 새로운 모델 생성")

history2 = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=16,
    verbose=0,
    validation_data=(X_val, Y_val),
    callbacks=[early_stop]
)

# 실제 값과 예측 값의 비교, 7 일치 평균 비교, 실제 값으로 복원
# predictions = model.predict(X_val)
# predictions_inv = invert_scale(predictions, scaler)
# Y_val_inv = invert_scale(Y_val, scaler)
# plt.figure(figsize=(10, 5))
# plt.title(f'{ticker} Stock Price Prediction')
# plt.plot(Y_val_inv.mean(axis=1), label='Actual Mean')
# plt.plot(predictions_inv.mean(axis=1), label='Predicted Mean')
# plt.xlabel(f'Next {forecast_horizon} days')
# plt.ylabel('Price')
# plt.legend()
# plt.show()



if old_history:
    combined_loss = list(old_history['loss']) + list(history2.history['loss'])
    combined_val_loss = list(old_history['val_loss']) + list(history2.history['val_loss'])
else:
    combined_loss = list(history2.history['loss'])
    combined_val_loss = list(history2.history['val_loss'])

# loss 그래프 그리기
# 0에 수렴하면 학습이 정상, 다시 오르면 과적합
plt.figure(figsize=(10, 5))
plt.plot(combined_loss, label='Train Loss')
plt.plot(combined_val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()



# **합쳐진 history 저장**
history_to_save = {
    'loss': combined_loss,
    'val_loss': combined_val_loss
}
with open(history_file, 'wb') as f:
    pickle.dump(history_to_save, f)