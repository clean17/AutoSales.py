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

from utils import create_multistep_dataset, fetch_stock_data, create_model_16, create_model_32, create_model_64,create_model_128, add_technical_features

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

'''
15/3이 예측 좋다?
400~410 정도가 좋아
'''
DATA_COLLECTION_PERIOD = 400
PREDICTION_PERIOD = 3
LOOK_BACK = 15

# 데이터 수집
tickers = ['006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

today = datetime.today().strftime('%Y%m%d')
# start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

for i in range(4):
    # period = DATA_COLLECTION_PERIOD + (10*i)
    # print('period : ', period)
    start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
    # start_date = (datetime.today() - timedelta(days=period)).strftime('%Y%m%d')
    # print('LOOK_BACK : ', (LOOK_BACK + i))

    total_mse = 0
    total_r2 = 0
    total_cnt = 0

    for count, ticker in enumerate(tickers):
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        data = add_technical_features(fetch_stock_data(ticker, start_date, today))

        threshold = 0.1  # 10%
        cols_to_drop = [
            col for col in data.columns
            if (data[col].isna().mean() > threshold) or (np.isinf(data[col]).mean() > threshold)
        ]
        if len(cols_to_drop) > 0:
            print("Drop candidates:", cols_to_drop)
            continue

        print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

        # 데이터 스케일링
        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_cols = [
            '종가', '고가', 'PBR', '저가',
            '거래량', 'RSI14',
            'ma10_gap',
        ]
        # feature_cols = [
        #     '종가', '고가', '저가', '거래량'
        # ]
        X_for_model = data[feature_cols].fillna(0) # 모델 feature NaN을 0으로
        scaled_data = scaler.fit_transform(X_for_model)

        # 시계열 데이터를 윈도우로 나누기
        X, Y = create_multistep_dataset(scaled_data, (LOOK_BACK + i), PREDICTION_PERIOD, 0)

        # 학습 데이터와 검증 데이터 분리
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, shuffle=False)
        if X_train.shape[0] < 50:
            print('샘플 부족 : ', X_train.shape)
            continue

        model = create_model_64((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD)

        # 콜백 설정
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
        history = model.fit(X_train, Y_train, batch_size=8, epochs=200,
                            validation_data=(X_val, Y_val),
                            shuffle=False, verbose=0, callbacks=[early_stop])


        # X_val : 검증에 쓸 여러 시계열 구간의 집합
        # predictions : 검증셋 각 구간(윈도우)에 대해 미래 PREDICTION_PERIOD만큼의 예측치 반환
        # shape: (검증샘플수, 예측일수)
        predictions = model.predict(X_val, verbose=0)
        # print('predictions (샘플, 예측일)', predictions.shape)

        # MSE
        mse = mean_squared_error(Y_val, predictions)
        # print("MSE:", mse)
        total_mse += mse

        # R-squared; (0=엉망, 1=완벽)
        r2 = r2_score(Y_val, predictions)
        # print("R-2:", r2)
        total_r2 += r2
        total_cnt += 1


    print('result1 : ', total_mse/total_cnt)
    print('result2 : ', total_r2/total_cnt)






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
  
0.01 이하면 매우 잘 맞추는 모델(스케일이 0~1로 정규화되어 있으면)
0.005 이하면 “상당히 우수”
0.001 이하면 “거의 오버핏/탁월”

R2
0.7~0.8: 실전에서 "꽤 우수한" 성능 (시계열/주가 등 변동성 큰 데이터 기준)
0.8~0.9: "아주 잘 맞추는" 예측
0.9 이상: 거의 완벽 (실전에서는 드물며, 오버피팅 의심도)
0.5~0.7: “적당히 쓸만한” 모델
0.5 이하: 실전 활용도 낮음 (정확도 개선 필요)
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
