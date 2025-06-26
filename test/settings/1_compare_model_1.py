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
import sys
import pickle
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data, compute_rsi, create_model_32, create_model_64,create_model_128, invert_scale

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


DATA_COLLECTION_PERIOD = 400
PREDICTION_PERIOD = 3
LOOK_BACK = 15
window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

# 데이터 수집
ticker = '443060' # 마린솔루션
ticker = '005690' # 파미셀
ticker = '000660' # 하이닉스
today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
data = fetch_stock_data(ticker, start_date, today)

data['MA5'] = data['종가'].rolling(window=5).mean()
data['MA10'] = data['종가'].rolling(window=10).mean()
data['MA5_slope'] = data['MA5'].diff() # diff() 차이르 계산하는 함수
data['MA10_slope'] = data['MA10'].diff()
data['MA15'] = data['종가'].rolling(window=15).mean()
data['MA20'] = data['종가'].rolling(window=window).mean()
data['STD20'] = data['종가'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])
# 이동평균 기울기(변화량)
data['MA5_slope'] = data['MA5'].diff() # diff() 차이르 계산하는 함수
data['MA15_slope'] = data['MA15'].diff()
# 볼린저밴드 위치 (현재가가 상단/하단 어디쯤?)
data['BB_perc'] = (data['종가'] - data['LowerBand']) / (data['UpperBand'] - data['LowerBand'] + 1e-9) # # 0~1 사이. 1이면 상단, 0이면 하단, 0.5면 중앙
# 거래량 증감률 >> 거래량이 0이거나, 직전 거래량이 0일 때 문제 발생
data['Volume_change'] = data['거래량'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
# RSI (14일)
data['RSI14'] = compute_rsi(data['종가'])


# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
# feature_cols = [
# #     '시가', '고가', '저가',
# #      '거래량',
#     '종가', 'MA15_slope', 'RSI14', 'BB_perc',
#     'Volume_change'
# ]
# feature_cols = [
#     '종가', '시가', '고가', '저가', '거래량'
# ]
# feature_cols = [
#     '고가', '저가', '종가', 'RSI14', 'BB_perc', 'Volume_change', 'MA15_slope'
# ]
feature_cols = [
    '종가', 'RSI14', 'BB_perc',
    'MA5_slope',
    '거래량',
#     'Volume_change'
#     'MA10_slope',
]
X_for_model = data[feature_cols].fillna(0) # 모델 feature NaN을 0으로
scaled_data = scaler.fit_transform(X_for_model)
# scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD, 0)

# 학습 데이터와 검증 데이터 분리
# random_state; 데이터를 랜덤하게 분할할 때의 랜덤 시드(seed) 값 > 항상 같은 방식으로 데이터를 나눔 (재현성 보장)
# val(test) set은 10%
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
print(X_train.shape)
# print(Y_train.shape)


model = create_model_128((X.shape[1], X.shape[2]), PREDICTION_PERIOD)

# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원

model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=8,
    verbose=0,
    validation_data=(X_val, Y_val),
    shuffle=False,
    callbacks=[early_stop]
)


# 예측 값 (predictions)과 실제 값 (y_val)
# print('x-val', len(X_val))
# print('y-val', len(Y_val))
predictions = model.predict(X_val)
# print("Y_val NaN 개수:", np.isnan(Y_val).sum())
# print("predictions NaN 개수:", np.isnan(predictions).sum())

# MSE
mse = mean_squared_error(Y_val, predictions)
print("Mean Squared Error:", mse)

# MAE
# mae = mean_absolute_error(Y_val, predictions)
# print("Mean Absolute Error:", mae)

# RMSE
# rmse = np.sqrt(mse)
# print("Root Mean Squared Error:", rmse)

# R-squared; (0=엉망, 1=완벽)
r2 = r2_score(Y_val, predictions)
print("R-squared:", r2)

'''
훈련용/검증용 loss

0.01~0.02 이하:
  → 매우 우수 (1~2% 미만 평균 오차)
0.02~0.05:
  → 실전 충분, 대부분의 실전 예측에서 이 정도면 OK
0.05~0.08:
  → 크다고 느껴질 수 있음, 신호가 “정확히 맞는지” 확인 필요
0.1 이상:
  → 오버피팅/과소적합/샘플 부족 가능성.
  실전에서는 예측 신호 품질 직접 확인 필요
'''

'''
LSTM/수집일/학습데이터/예측일/샘플수/배치/레이어/mse/squared

'종가', 'MA15_slope', 'RSI14', 'BB_perc', 'Volume_change'
32 / 100 / 15 / 3 / 45 / 16 - 0.007 0.65
32 / 120 / 15 / 3 / 57 / 16 - 0.021 0.71
32 / 140 / 15 / 3 / 70 / 16 - 0.018 0.64
32 / 160 / 15 / 3 / 79 / 16 - 0.023 0.65
32 / 200 / 15 / 3 / 102 / 16 - 0.021 0.60
32 / 400 / 15 / 3 / 224 / 16 - 0.022 0.59

32 / 120 / 15 / 3 / 57 / 8 - 0.020 0.73
32 / 130 / 15 / 3 / 63 / 8 - 0.012 0.78
32 / 135 / 15 / 3 / 67 / 8 - 0.012 0.84
32 / 140 / 15 / 3 / 70 / 8 - 0.003 0.92 ##########
32 / 150 / 15 / 3 / 73 / 8 - 0.020 0.73
32 / 160 / 15 / 3 / 79 / 8 - 0.020 0.69

32 / 120 / 15 / 3 / 57 / 4 - 0.020 0.72
32 / 140 / 15 / 3 / 70 / 4 - 0.003 0.92 ###########
32 / 160 / 15 / 3 / 79 / 4 - 0.020 0.69
32 / 200 / 15 / 3 / 102 / 4 - 0.021 0.60
32 / 400 / 15 / 3 / 224 / 4 - 0.012 0.77

64 / 140 / 15 / 3 / 70 / 8 - 0.004 0.90
128 / 140 / 15 / 3 / 70 / 4 - 0.002 0.94 ##########
128 / 140 / 15 / 3 / 70 / 8 - 0.002 0.95 ##########
128 / 140 / 15 / 3 / 70 / 16 - 0.002 0.95 ##########

32 / 140 / 15 / 3 / 70 / 8
'시가', '고가', '저가', '종가', '거래량'
  - 0.006 0.90
'시가', '고가', '저가', '종가', '거래량', 'RSI14'
  - 0.006 0.90
'시가', '고가', '저가', '종가', '거래량', 'RSI14', 'BB_perc'
  - 0.005 0.91
'시가', '고가', '저가', '종가', '거래량', 'RSI14', 'BB_perc', 'Volume_change'
  - 0.005 0.91
'시가', '고가', '저가', '종가', 'RSI14', 'BB_perc', 'Volume_change'
  - 0.004 0.93
'고가', '저가', '종가', 'RSI14', 'BB_perc', 'Volume_change'
  - 0.003 0.91
'고가', '저가', '종가', 'RSI14', 'BB_perc', 'Volume_change', 'MA15_slope'
  - 0.002 0.94
'종가', '고가', '저가', 'RSI14', 'BB_perc', 'Volume_change', 'MA15_slope'
  - 0.003 0.95
'''



"""
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
"""
