import os
import pandas as pd
import numpy as np
import glob
import re
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set random seed for reproducibility
# tf.random.set_seed(42)
DROPOUT = 0.3
# 종목
ticker = '036460' # 한국가스공사
# ticker = '009470' # 삼화전기
# ticker = '003960' # 사조대림
# ticker = '007160' # 사조산업
# ticker = '249420' # 일동제약
ticker = '373200'
# 예측 기간
PREDICTION_PERIOD = 10
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365

# EarlyStopping
EARLYSTOPPING_PATIENCE = 20
# 데이터셋 크기 ( 타겟 3일: 20, 5-7일: 30~50, 10일: 40~60, 15일: 50~90)
LOOK_BACK = 60
# 반복 횟수 ( 5일: 100, 7일: 150, 10일: 200, 15일: 300)
EPOCHS_SIZE = 200
BATCH_SIZE = 32

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


# output_dir = 'D:\\stocks'
output_dir = 'D:\\kospi_stocks'
# model_dir = os.path.join(output_dir, 'models')
# model_dir = 'kospi_30_models'
model_dir = 'models'
model_dir = 'kospi_kosdaq_60(10)_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    data = ohlcv[['종가', '저가', '고가', '거래량']]
    return data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
    for i in range(len(dataset) - look_back):
        # X.append(dataset[i:i+look_back])   # 모든 열을 사용
        X.append(dataset[i:i+look_back, :])  # 모든 열을 사용
        Y.append(dataset[i+look_back, 0])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    # model = Sequential([
    #     LSTM(256, return_sequences=True, input_shape=input_shape),
    #     LSTM(128, return_sequences=True),
    #     LSTM(64, return_sequences=False),
    #     Dense(128),
    #     Dense(64),
    #     Dense(32),
    #     Dense(1)
    # ])

    model = tf.keras.Sequential()
    # model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    # model.add(Dropout(DROPOUT))
    #
    # # 두 번째 LSTM 층
    # model.add(LSTM(256, return_sequences=True))
    # model.add(Dropout(DROPOUT))
    #
    # # 세 번째 LSTM 층
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(DROPOUT))
    #
    # # 네 번째 LSTM 층
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dropout(DROPOUT))
    #
    # # Dense 레이어
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(DROPOUT))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(DROPOUT))
    # model.add(Dense(32, activation='relu'))

    model.add((LSTM(256, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(DROPOUT))

    # 두 번째 LSTM 층
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT))

    # 세 번째 LSTM 층
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(DROPOUT))

    # Dense 레이어
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(16, activation='relu'))

    # 출력 레이어
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.compile(optimizer='rmsprop', loss='mse')
    return model


stock_name = stock.get_market_ticker_name(ticker)

# 데이터 로드 및 스케일링
data = fetch_stock_data(ticker, start_date, today)

# 마지막 행의 데이터를 가져옴
last_row = data.iloc[-1]
# 종가가 0.0인지 확인
if last_row['종가'] == 0.0:
    print("종가가 0 이므로 작업을 건너뜁니다.")

print(last_row['종가'])

todayTime = datetime.today()  # `today`를 datetime 객체로 유지

# 3달 전의 종가와 비교
three_months_ago_date = todayTime - pd.DateOffset(months=3)
data_before_three_months = data.loc[:three_months_ago_date]


if len(data_before_three_months) > 0:
    closing_price_three_months_ago = data_before_three_months.iloc[-1]['종가']
    print(f' 3달전 종가의 30% 하락: {closing_price_three_months_ago * 0.7}')
    if closing_price_three_months_ago > 0 and (last_row['종가'] < closing_price_three_months_ago * 0.7):
        print(f" 최근 종가가 3달 전의 종가보다 30% 이상 하락했으므로 작업을 건너뜁니다.")

# 1년 전의 종가와 비교
one_year_ago_date = todayTime - pd.DateOffset(days=365)
data_before_one_year = data.loc[:one_year_ago_date]

# 1년 전과 비교
if len(data_before_one_year) > 0:
    closing_price_one_year_ago = data_before_one_year.iloc[-1]['종가']
    print(f' 1년전 종가의 50% 하락 : {closing_price_one_year_ago * 0.5}')
    if closing_price_one_year_ago > 0 and (last_row['종가'] < closing_price_one_year_ago * 0.5):
        print(f" 최근 종가가 1년 전의 종가보다 50% 이상 하락했으므로 작업을 건너뜁니다.")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# 데이터셋 생성
# 30일 구간의 데이터셋, (365 - 30 + 1)-> 336개의 데이터셋

X, Y = create_dataset(scaled_data, LOOK_BACK)

# 난수 데이터셋 분할
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)


model_file_path = os.path.join(model_dir, f'{ticker}_model_v2.Keras')
if os.path.exists(model_file_path):
    print('get_save_models ...')
    model = tf.keras.models.load_model(model_file_path)
else:
    # 단순히 모델만 생성
    model = create_model((X_train.shape[1], X_train.shape[2]))
    # 지금은 매번 학습할 예정이다
    # model.fit(X, Y, epochs=3, batch_size=32, verbose=1, validation_split=0.1)
    # model.save(model_file_path)


# 모델 생성
# model = create_model((X_train.shape[1], X_train.shape[2]))

# 조기 종료 설정
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=EARLYSTOPPING_PATIENCE,  # 10 에포크 동안 개선 없으면 종료
    verbose=0,
    mode='min',
    restore_best_weights=True  # 최적의 가중치를 복원
)

# 입력 X에 대한 예측 Y 학습
# model.fit(X, Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1) # verbose=1 은 콘솔에 진척도
# 모델 학습
model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=1,
          validation_data=(X_val, Y_val), callbacks=[early_stopping])

model.save(model_file_path) # 체크포인트가 자동으로 최적의 상태를 저장한다

close_scaler = MinMaxScaler()
close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

# 예측, 입력 X만 필요하다
predictions = model.predict(X[-PREDICTION_PERIOD:])
predicted_prices = close_scaler.inverse_transform(predictions).flatten()
# model.save(model_file_path)

last_close = data['종가'].iloc[-1]
future_return = (predicted_prices[-1] / last_close - 1) * 100

extended_prices = np.concatenate((data['종가'].values, predicted_prices))
extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
last_price = data['종가'].iloc[-1]

plt.figure(figsize=(16, 8))
plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
plt.title(f'{ticker} - Actual vs Predicted Prices {today} {stock_name} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
# plt.show()

# 디렉토리 내 파일 검색 및 삭제
for file_name in os.listdir(output_dir):
    if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
        print(f"Deleting existing file: {file_name}")
        os.remove(os.path.join(output_dir, file_name))

final_file_name = f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ].png'
final_file_path = os.path.join(output_dir, final_file_name)

plt.savefig(final_file_path)
plt.close()