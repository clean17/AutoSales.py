from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
pickle_dir = os.path.join(BASE_DIR, 'pickle')

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, get_kor_ticker_dict_list, \
    drop_trading_halt_rows, fetch_stock_data

# 데이터 수집
ticker = '000660' # 하이닉스
filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
data = pd.read_pickle(filepath)

# 2차 생성 feature
data = add_technical_features(data)

feature_cols = [
    '시가', '저가', '고가', '종가',
    'Vol_logdiff',
    # 'MA5_slope',
]

# NaN/inf 정리
data = data.replace([np.inf, -np.inf], np.nan)
# feature_cols만 남기고 dropna
feature_cols = [c for c in feature_cols if c in data.columns]
X_df = data.dropna(subset=feature_cols).loc[:, feature_cols]

idx_close = feature_cols.index('종가')

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))


scaler_X = MinMaxScaler().fit(X_df)          # 또는 ColumnTransformer 등
scaled_data = scaler_X.fit_transform(X_df)
scaler_y = MinMaxScaler().fit(scaled_data[:, [idx_close]])
# 시계열 데이터를 윈도우로 나누기
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

idx_close = feature_cols.index('종가')
look_back = 15

X, Y = create_dataset(scaled_data, look_back)
# X, Y = create_multistep_dataset(scaled_data, look_back, 1, idx_close)
print(X.shape) # (272, 15, 34)
print(Y.shape) # (272,)

# 학습 데이터와 테스트 데이터 분리
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

'''
Dense 레이어의 역할
모든 입력 뉴런이 모든 출력 뉴런과 연결되어 있는 층
복잡한 패턴을 비선형적으로 조합해서 결과값(예측, 분류, 회귀 등)을 만든다.

Dense(32) → Dense(16) → Dense(1): 복잡한 문제, 더 높은 표현력 필요할 때

Dense(32) → Dense(1): 간단하거나 빠르게 실험할 때

Dropout 0.2/0.3 비교 테스트 필요
'''
# 모델 생성
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

# model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, Y_train, batch_size=16, epochs=200, verbose=0, shuffle=False, validation_data=(X_test, Y_test), callbacks=[early_stop])

'''
Train Actual (훈련 실제 값):
- 훈련 데이터에 대한 실제 종가 값을 나타냅니다.
- 주식 데이터의 훈련 기간 동안의 실제 주가 변동을 보여줍니다.

Test Actual (테스트 실제 값):
- 테스트 데이터에 대한 실제 종가 값을 나타냅니다.
- 주식 데이터의 테스트 기간 동안의 실제 주가 변동을 보여줍니다.
- 그래프에서 Y_train_actual 데이터 뒤에 이어져서 나타납니다.

Train Predict (훈련 예측 값):
- 훈련 데이터에 대한 모델의 예측 값을 나타냅니다.
- 모델이 훈련 데이터로 학습한 결과를 보여줍니다.

Test Predict (테스트 예측 값):
- 테스트 데이터에 대한 모델의 예측 값을 나타냅니다.
- 모델이 새로운 데이터(테스트 데이터)에서 예측한 결과를 보여줍니다.
- 그래프에서 Y_train_actual 데이터 뒤에 이어져서 나타납니다.

Train Actual과 Train Predict과 매우 유사하면 잘 학습되었다.
'''
# 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 역변환
train_predict_inv = scaler_y.inverse_transform(train_predict.reshape(-1,1)).ravel()
test_predict_inv  = scaler_y.inverse_transform(test_predict.reshape(-1,1)).ravel()
Y_train_actual    = scaler_y.inverse_transform(Y_train.reshape(-1,1)).ravel()
Y_test_actual     = scaler_y.inverse_transform(Y_test.reshape(-1,1)).ravel()

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(Y_train_actual, label='Train Actual')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(Y_test_actual)), Y_test_actual, label='Test Actual')
plt.plot(train_predict_inv, label='Train Predict')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(test_predict_inv)), test_predict_inv, label='Test Predict')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
