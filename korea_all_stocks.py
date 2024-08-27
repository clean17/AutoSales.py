import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash

# 과적합 방지
DROPOUT = 0.3
# 시작 종목 인덱스 ( 중단된 경우 다시 시작용 )
count = 0
# 예측 기간
PREDICTION_PERIOD = 10
# 예측 성장률
EXPECTED_GROWTH_RATE = 3
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365
# EarlyStopping
EARLYSTOPPING_PATIENCE = 20
# 데이터셋 크기 ( 타겟 3일: 20, 5-7일: 30~50, 10일: 40~60, 15일: 50~90)
LOOK_BACK = 60
# 반복 횟수 ( 5일: 100, 7일: 150, 10일: 200, 15일: 300)
EPOCHS_SIZE = 100
BATCH_SIZE = 32

AVERAGE_VOLUME = 20000
AVERAGE_TRADING_VALUE = 1000000000

# 그래프 저장 경로
output_dir = 'D:\\kospi_stocks'
# 모델 저장 경로
model_dir = 'kospi_kosdaq_60(10)_models' # 신규모델

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

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
        X.append(dataset[i:i+look_back, :])
        Y.append(dataset[i+look_back, 0])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = tf.keras.Sequential()

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

    model.compile(optimizer='adam', loss='mean_squared_error') # mes
    # model.compile(optimizer='rmsprop', loss='mse')
    return model




# 초기 설정
max_iterations = 5
all_tickers = stock.get_market_ticker_list(market="KOSPI") + stock.get_market_ticker_list(market="KOSDAQ")
# 종목 코드와 이름 딕셔너리 생성
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in all_tickers}


for iteration in range(max_iterations):
    print(f"==== Iteration {iteration + 1}/{max_iterations} ====")

    # 디렉토리 내 파일 검색 및 휴지통으로 보내기
    for file_name in os.listdir(output_dir):
        if file_name.startswith(today):
            # print(f"Sending to trash: {file_name}")
            send2trash(os.path.join(output_dir, file_name))

    if iteration == 0:
        tickers = all_tickers  # 첫 번째 반복은 모든 종목
    else:
        tickers = saved_tickers  # 그 이후는 이전 반복에서 저장된 종목들

    # 결과를 저장할 배열
    saved_tickers = []


    for count, ticker in enumerate(tickers):
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        print(f"Processing {count+1}/{len(tickers)} : {stock_name} {ticker}")

        data = fetch_stock_data(ticker, start_date, today)

        # 마지막 행의 데이터를 가져옴
        last_row = data.iloc[-1]
        # 종가가 0.0이거나 400원 미만인지 확인
        if last_row['종가'] == 0.0 or last_row['종가'] < 400:
            print("                                                        종가가 0이거나 400원 미만이므로 작업을 건너뜁니다.")
            continue

        # 데이터가 충분하지 않으면 건너뜀
        if data.empty or len(data) < LOOK_BACK:
            print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
            continue

        # 일일 평균 거래량
        average_volume = data['거래량'].mean() # volume
        if average_volume <= AVERAGE_VOLUME:
            print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
            continue

        # 일일 평균 거래대금
        trading_value = data['거래량'] * data['종가']
        average_trading_value = trading_value.mean()
        if average_trading_value <= AVERAGE_TRADING_VALUE:
            formatted_value = f"{average_trading_value / 100000000:.0f}억"
            print(f"                                                        평균 거래액({formatted_value})이 부족하여 작업을 건너뜁니다.")
            continue

        todayTime = datetime.today()  # `today`를 datetime 객체로 유지

        # 3달 전의 종가와 비교
        three_months_ago_date = todayTime - pd.DateOffset(months=3)
        data_before_three_months = data.loc[:three_months_ago_date]

        # 1년 전의 종가와 비교
        # 데이터를 기준으로 반복해서 날짜를 줄여가며 찾음
        data_before_one_year = pd.DataFrame()  # 초기 빈 데이터프레임
        days_offset = 365

        while days_offset >= 360:
            one_year_ago_date = todayTime - pd.DateOffset(days=days_offset)
            data_before_one_year = data.loc[:one_year_ago_date]

            if not data_before_one_year.empty:  # 빈 배열이 아닌 경우
                break  # 조건을 만족하면 반복 종료
            days_offset -= 1  # 다음 날짜 시도

        # 두 조건을 모두 만족하는지 확인
        should_skip = False

        if len(data_before_three_months) > 0 and len(data_before_one_year) > 0:
            closing_price_three_months_ago = data_before_three_months.iloc[-1]['종가']
            closing_price_one_year_ago = data_before_one_year.iloc[-1]['종가']

            if (closing_price_three_months_ago > 0 and last_row['종가'] < closing_price_three_months_ago * 0.7) and \
                    (closing_price_one_year_ago > 0 and last_row['종가'] < closing_price_one_year_ago * 0.5):
                should_skip = True

        if should_skip:
            print(f"                                                        최근 종가가 3달 전의 종가보다 30% 이상 하락하고 1년 전의 종가보다 50% 이상 하락했으므로 작업을 건너뜁니다.")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X, Y = create_dataset(scaled_data, LOOK_BACK)

        if len(X) < 2 or len(Y) < 2:
            print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다.")
            continue

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

        model_file_path = os.path.join(model_dir, f'{ticker}_model_v2.Keras')
        if os.path.exists(model_file_path):
            model = tf.keras.models.load_model(model_file_path)
        else:
            model = create_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLYSTOPPING_PATIENCE,  # 지정한 에포크 동안 개선 없으면 종료
            verbose=0,
            mode='min',
            restore_best_weights=True  # 최적의 가중치를 복원
        )

        # 모델 학습
        model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=0, # 충분히 모델링 되었으므로 20번만
                  validation_data=(X_val, Y_val), callbacks=[early_stopping]) # 체크포인트 자동저장

        close_scaler = MinMaxScaler()
        close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

        # 예측, 입력 X만 필요하다
        predictions = model.predict(X[-PREDICTION_PERIOD:])
        predicted_prices = close_scaler.inverse_transform(predictions).flatten()

        model.save(model_file_path)

        last_close = data['종가'].iloc[-1]
        future_return = (predicted_prices[-1] / last_close - 1) * 100

        # 성장률 이상만
        if future_return < EXPECTED_GROWTH_RATE:
            continue

        extended_prices = np.concatenate((data['종가'].values, predicted_prices))
        extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
        last_price = data['종가'].iloc[-1]

        plt.figure(figsize=(16, 8))
        plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
        plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
        plt.title(f'{today_us} {ticker} {stock_name} [ {last_price:.2f} ] (Expected Return: {future_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)

        # 디렉토리 내 파일 검색 및 삭제
        for file_name in os.listdir(output_dir):
            if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
                print(f"Deleting existing file: {file_name}")
                os.remove(os.path.join(output_dir, file_name))

        final_file_name = f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price:.2f} ].png'
        final_file_path = os.path.join(output_dir, final_file_name)
        print(final_file_name)
        plt.savefig(final_file_path)
        plt.close()

        saved_tickers.append(ticker)

    print("Files were saved for the following tickers:")
    print(saved_tickers)

    for file_name in os.listdir(output_dir):
        if file_name.startswith(today):
            print(f"{file_name}")