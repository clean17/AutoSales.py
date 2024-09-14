import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash

# Set random seed for reproducibility
# tf.random.set_seed(42)

DROPOUT = 0.3
DROPOUT = 0.3
PREDICTION_PERIOD = 5
EXPECTED_GROWTH_RATE = 2
DATA_COLLECTION_PERIOD = 180
EARLYSTOPPING_PATIENCE = 10
LOOK_BACK = 30
EPOCHS_SIZE = 100
BATCH_SIZE = 32
AVERAGE_VOLUME = 30000
AVERAGE_TRADING_VALUE = 2000000000

# 미국 동부 시간대 설정
us_timezone = pytz.timezone('America/New_York')
now_us = datetime.now(us_timezone)
# 현재 시간 출력
today_us = now_us.strftime('%Y-%m-%d %H:%M:%S')
print("미국 동부 시간 기준 현재 시각:", today_us)
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

today_us = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')
# start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')


# S&P 500 종목을 가져오는 함수
def get_sp500_tickers():
    sp500_constituents = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]  # Wikipedia에서 종목 가져오기
    sp500_tickers = sp500_constituents['Symbol'].tolist()  # 티커 리스트 추출
    return sp500_tickers

# 나스닥 100 종목을 가져오는 함수
def get_nasdaq100_tickers():
    nasdaq100_constituents = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]  # Wikipedia에서 종목 가져오기
    nasdaq100_tickers = nasdaq100_constituents['Ticker'].tolist()  # 티커 리스트 추출
    return nasdaq100_tickers

# 다우존스 30 종목을 가져오는 함수
def get_dowjones_tickers():
    dowjones_constituents = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[1]  # Wikipedia에서 종목 가져오기
    dowjones_tickers = dowjones_constituents['Symbol'].tolist()  # 티커 리스트 추출
    return dowjones_tickers

# 나스닥 100 종목 중 S&P 500에 속하지 않는 다우존스 종목을 구하는 함수
def get_all_unique_tickers():
    # S&P 500, 나스닥 100, 다우존스 종목 리스트 가져오기
    sp500_tickers = get_sp500_tickers()  # S&P 500 종목 리스트
    nasdaq100_tickers = get_nasdaq100_tickers()  # 나스닥 100 종목 리스트
    dowjones_tickers = get_dowjones_tickers()  # 다우존스 30 종목 리스트

    # 세 지수의 모든 종목을 합치고 중복 제거
    all_tickers = set(sp500_tickers + nasdaq100_tickers + dowjones_tickers)

    # 결과 출력
    # print('모든 지수의 중복 없는 종목 리스트:', all_tickers)

    return list(all_tickers)


tickers = get_all_unique_tickers()

output_dir = 'D:\\sp500'
model_dir = 'sp_30(5)180_rmsprop_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터를 가져오는 함수
# def fetch_stock_data(ticker, fromdate, todate):
#     stock_data = yf.download(ticker, start=fromdate, end=todate)
#     # stock_data = fdr.DataReader(ticker, start=fromdate, end=todate)
#     if stock_data.empty:
#         return pd.DataFrame()
#     # 선택적인 컬럼만 추출하고 NaN 값을 0으로 채움
#     stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
#     stock_data['PER'] = 0  # FinanceDataReader의 PER 필드 대체
#     return stock_data

def fetch_stock_data(ticker, fromdate, todate):
    stock_data = yf.download(ticker, start=fromdate, end=todate)
    if stock_data.empty:
        return pd.DataFrame()

    # yfinance를 통해 주식 정보 가져오기
    stock_info = yf.Ticker(ticker).info

    # PER 값을 info에서 추출, 없는 경우 0으로 처리
    per_value = stock_info.get('trailingPE', 0)  # trailingPE를 사용하거나 없으면 0

    # PBR 값을 info에서 추출, 없는 경우 0으로 처리
    pbr_value = stock_info.get('priceToBook', 0)

    # 주식 데이터에 PER 컬럼 추가
    stock_data['PER'] = per_value
    stock_data['PBR'] = pbr_value

    # 선택적인 컬럼 추출 및 NaN 값 처리
    stock_data = stock_data[['Close', 'High', 'Low', 'Volume', 'PER', 'PBR']].fillna(0)
    return stock_data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        Y.append(dataset[i+look_back, 0])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add((LSTM(256, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(DROPOUT))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='rmsprop', loss='mse')
    return model



# @tf.function(reduce_retracing=True)
# def predict_model(model, data):
#     return model(data)

# 반복 설정
max_iterations = 5
# specific_tickers = ['ABT', 'ALB', 'CHTR', 'DAY', 'DG', 'EL', 'INTC', 'MTCH', 'NSC', 'STLD', 'SMCI', 'UAL', 'WBA', 'WBD', 'WFC', 'TEAM']

ticker_returns = {} # 평균
saved_tickers = [] # 회차 저장

for iteration in range(max_iterations):
    print("\n")
    print(f"==== Iteration {iteration + 1}/{max_iterations} ====")

    for file_name in os.listdir(output_dir):
        if file_name.startswith(today):
            # print(f"Sending to trash: {file_name}")
            send2trash(os.path.join(output_dir, file_name))

    # 특정 배열을 가져왔을때 / 예를 들어 60(10)으로 가져온 배열을 40(5)로 돌리는 경우
    if iteration != 0:
        tickers = saved_tickers  # 2회차 부터 이전 반복에서 저장된 종목들

    # 결과를 저장할 배열
    saved_tickers = []

    # for ticker in tickers:
    for count, ticker in enumerate(tickers):
        print(f"Processing {count+1}/{len(tickers)} : {ticker}")
        # count += 1
        data = fetch_stock_data(ticker, start_date_us, today_us)

        if data.empty or len(data) < LOOK_BACK:  # 데이터가 충분하지 않으면 건너뜀
            print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
            continue

        last_row = data.iloc[-1]
        if last_row['Close'] == 0.0:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X, Y = create_dataset(scaled_data, LOOK_BACK)

        # Convert the scaled_data to a TensorFlow tensor
        # scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)

        # X, Y = create_dataset(scaled_data_tensor.numpy(), LOOK_BACK)  # numpy()로 변환하여 create_dataset 사용
        # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        if len(X) < 2 or len(Y) < 2:
            print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다.")
            continue

        # 난수 데이터셋 분할
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

        model_file_path = os.path.join(model_dir, f'{ticker}_model_v2.Keras')
        if os.path.exists(model_file_path):
            model = tf.keras.models.load_model(model_file_path)
            if model.input_shape != (None, X_train.shape[1], X_train.shape[2]):
                print('Loaded model input shape does not match data input shape. Creating a new model.')
                model = create_model((X_train.shape[1], X_train.shape[2]))
        else:
            model = create_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLYSTOPPING_PATIENCE,  # 10 에포크 동안 개선 없으면 종료
            verbose=0,
            mode='min',
            restore_best_weights=True  # 최적의 가중치를 복원
        )

        model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=0,
                  validation_data=(X_val, Y_val), callbacks=[early_stopping])

        close_scaler = MinMaxScaler()
        close_prices_scaled = close_scaler.fit_transform(data[['Close']].values)

        # Make predictions with the model
        # predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))
        # predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()

        # 예측, 입력 X만 필요하다
        predictions = model.predict(X[-PREDICTION_PERIOD:])
        predicted_prices = close_scaler.inverse_transform(predictions).flatten()

        model.save(model_file_path)

        last_close = data['Close'].iloc[-1]
        future_return = (predicted_prices[-1] / last_close - 1) * 100

        if future_return < EXPECTED_GROWTH_RATE:
            continue

        average_volume = data['Volume'].mean() # volume
        if average_volume <= AVERAGE_VOLUME:
            print(f'                                                        average_volume ', average_volume)
            continue

        # 일일 평균 거래대금
        trading_value = data['Volume'] * data['Close']
        average_trading_value = trading_value.mean()
        if average_trading_value <= AVERAGE_TRADING_VALUE:
            formatted_value = f"{average_trading_value / 100000000:.0f}억"
            print(f'                                                        average_trading_value ', {formatted_value})
            continue

        if ticker in ticker_returns:
            ticker_returns[ticker].append(future_return)
        else:
            ticker_returns[ticker] = [future_return]

        saved_tickers.append(ticker)

        extended_prices = np.concatenate((data['Close'].values, predicted_prices))
        extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
        last_price = data['Close'].iloc[-1]

        plt.figure(figsize=(16, 8))
        plt.plot(extended_dates[:len(data['Close'].values)], data['Close'].values, label='Actual Prices', color='blue')
        plt.plot(extended_dates[len(data['Close'].values)-1:], np.concatenate(([data['Close'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
        plt.title(f'{ticker} {today_us} [ {last_price:.2f} ] (Expected Return: {future_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)

        # 디렉토리 내 파일 검색 및 삭제
        for file_name in os.listdir(output_dir):
            if file_name.startswith(f"{today}") and ticker in file_name:
                print(f"Deleting existing file: {file_name}")
                os.remove(os.path.join(output_dir, file_name))

        # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {ticker} [ {last_price:.2f} ].png')
        plt.savefig(file_path)
        plt.close()


    print("Files were saved for the following tickers:")
    print(saved_tickers)


results = []

for ticker in saved_tickers:
    if len(ticker_returns.get(ticker, [])) == 5:
        avg_future_return = sum(ticker_returns[ticker]) / 5
        results.append((avg_future_return, ticker)) # 튜플

# avg_future_return을 기준으로 내림차순 정렬
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, ticker in results:
    print(f"==== [ {avg_future_return:.2f}% ] {ticker} ====")