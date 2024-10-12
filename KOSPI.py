import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash
import ast

# Set random seed for reproducibility
# tf.random.set_seed(42)

specific_tickers = ['229640', '181710', '011780', '004270', '000490', '004835', '192650', '005935', '068290', '004430', '097520', '077500', '007110', '033250', '003350', '105630', '003280', '036120', '255220', '067160', '289080', '098460', '253590', '213420', '131970', '281740', '235980', '049950', '307870', '032850', '419120', '000250', '378800', '294630', '046890', '049180', '328380', '215600', '456010', '059120', '109610', '203400', '021080', '356680', '039200', '080580', '273060', '240810', '101160', '101730', '123420', '303530', '462350', '123570', '033230', '095700', '452300', '263700', '432470', '102370', '033540', '388870', '234100', '053610', '163730', '450330', '149980', '053300', '430690', '054920', '084990']

'''
1: 365 5 1차 필터링
2: 180 5 2차 필터링
3: 1차 > 2차 필터링
'''

# 조건 입력
CONDITION = int(input("CONDITION에 넣을 값을 입력하세요 (1: 전체 스캔, 2: 집중 스캔, 3: 후속 작업) : "))


# 공통 설정값
DROPOUT = 0.3
PREDICTION_PERIOD = 5
LOOK_BACK = 30
# 데이터셋 크기 ( 타겟 3일: 20, 5-7일: 30~50, 10일: 40~60, 15일: 50~90)
BATCH_SIZE = 32
# 그래프 저장 경로
output_dir = 'D:\\kospi_stocks'
# model_dir = os.path.join(output_dir, 'models')
os.makedirs(output_dir, exist_ok=True)
# 평균거래량
AVERAGE_VOLUME = 25000
# 평균거래대금
AVERAGE_TRADING_VALUE = 1800000000
MAX_ITERATIONS = 5

# 변수 초기화
tickers = None
model_dir = ''
saved_tickers = []  # 조건 만족한 종목을 저장할 리스트

if CONDITION == 1 or CONDITION == 3:
    # 예측 성장률 (기본값 : 5)
    EXPECTED_GROWTH_RATE = 5
    # 데이터 수집 기간
    DATA_COLLECTION_PERIOD = 365
    # EarlyStopping
    EARLYSTOPPING_PATIENCE = 5 #숏 10, 롱 20
    # 반복 횟수 ( 5일: 100, 7일: 150, 10일: 200, 15일: 300)
    EPOCHS_SIZE = 40
    # 모델 저장 경로
    model_dir = 'kospi_kosdaq_30(5)365_rmsprop_models_128'
    # 종목
    tickers_kospi = stock.get_market_ticker_list(market="KOSPI")
    tickers_kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
    tickers = tickers_kospi + tickers_kosdaq # 전체
    if CONDITION == 3:
        MAX_ITERATIONS = 6
elif CONDITION == 2:
    specific_tickers = input("specific_tickers에 넣을 값을 입력하세요 : ")
    specific_tickers = ast.literal_eval(specific_tickers)
    EXPECTED_GROWTH_RATE = 5
    DATA_COLLECTION_PERIOD = 180
    EARLYSTOPPING_PATIENCE = 10
    EPOCHS_SIZE = 100
    model_dir = 'kospi_kosdaq_30(5)180_rmsprop_models'
    tickers = None # 선택한 배열
else:
    print('잘못된 값을 입력했습니다.')
    exit()


today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
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

    if CONDITION == 1:
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
    elif CONDITION == 2:
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
    # model.compile(optimizer='adam', loss='mean_squared_error') # mes

    '''
    학습률을 각 파라미터에 맞게 조정하는 방식에서, 평균 제곱을 기반으로 학습률을 조정합니다. 
    특히, 시계열 데이터와 같이 그라디언트가 빠르게 변하는 경우에 잘 작동합니다.
    
    초기 실험 단계에서는 Adam을 사용하는 것을 추천드립니다. 
    Adam은 많은 경우에서 좋은 성능을 보이며, 하이퍼파라미터 튜닝 없이도 비교적 안정적인 학습을 제공합니다. 
    그 후, 모델의 성능을 RMSprop과 비교하여 어떤 옵티마이저가 주어진 데이터셋과 모델 구조에서 더 나은 결과를 제공하는지 평가해보는 것이 좋습니다.
    '''
    model.compile(optimizer='rmsprop', loss='mse')
    return model

# @tf.function(reduce_retracing=True)
# def predict_model(model, data):
#     return model(data)


if tickers is None:
    tickers = specific_tickers

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}
# 성장률을 저장할 튜플
ticker_returns = {}

for iteration in range(MAX_ITERATIONS):
    print("\n")
    print(f"==== Iteration {iteration + 1}/{MAX_ITERATIONS} ====")

    # 디렉토리 내 파일 검색 및 휴지통으로 보내기
    for file_name in os.listdir(output_dir):
        if file_name.startswith(today):
            # print(f"Sending to trash: {file_name}")
            send2trash(os.path.join(output_dir, file_name))

    # 특정 배열을 가져왔을때 / 예를 들어 60(10)으로 가져온 배열을 40(5)로 돌리는 경우
    if iteration != 0:
        tickers = saved_tickers  # 2회차 부터 이전 반복에서 저장된 종목들
        if CONDITION == 3:
            CONDITION = 2
            EXPECTED_GROWTH_RATE = 5
            DATA_COLLECTION_PERIOD = 180
            EARLYSTOPPING_PATIENCE = 10
            EPOCHS_SIZE = 100
            model_dir = 'kospi_kosdaq_30(5)180_rmsprop_models'

    # 결과를 저장할 배열
    saved_tickers = []


    # for ticker in tickers[count:]:
    for count, ticker in enumerate(tickers):
    # for ticker in tickers[count:count+1]:
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        print(f"Processing {count+1}/{len(tickers)} : {stock_name} {ticker}")

        data = fetch_stock_data(ticker, start_date, today)

        # 마지막 행의 데이터를 가져옴
        last_row = data.iloc[-1]
        # 종가가 0.0이거나 400원 미만인지 확인
        if last_row['종가'] == 0.0 or last_row['종가'] < 500:
            print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
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

        is_three_month_skip = False
        if len(data_before_three_months) > 0:
            closing_price_three_months_ago = data_before_three_months.iloc[-1]['종가']
            if closing_price_three_months_ago > 0 and (last_row['종가'] < closing_price_three_months_ago * 0.72): # 30~40
                print(f"                                                        최근 종가가 3달 전의 종가보다 28% 이상 하락했으므로 작업을 건너뜁니다.")
                is_three_month_skip = True
                continue

        if DATA_COLLECTION_PERIOD >= 360:
            # 1년 전의 종가와 비교, 데이터를 기준으로 반복해서 날짜를 줄여가며 찾음
            data_before_one_year = pd.DataFrame()  # 초기 빈 데이터프레임
            days_offset = 365

            while days_offset >= 360:
                one_year_ago_date = todayTime - pd.DateOffset(days=days_offset)
                data_before_one_year = data.loc[:one_year_ago_date]
                if not data_before_one_year.empty:  # 빈 배열이 아닌 경우
                    break  # 조건을 만족하면 반복 종료
                days_offset -= 1  # 다음 날짜 시도

            is_one_year_skip = False
            # 1년 전과 비교
            if len(data_before_one_year) > 0:
                closing_price_one_year_ago = data_before_one_year.iloc[-1]['종가']
                if closing_price_one_year_ago > 0 and (last_row['종가'] < closing_price_one_year_ago * 0.55):
                    print(f"                                                        최근 종가가 1년 전의 종가보다 45% 이상 하락했으므로 작업을 건너뜁니다.")
                    is_one_year_skip = True
                    continue

            # 두 조건을 모두 만족하는지 확인
            if (is_three_month_skip and is_one_year_skip):
                print(f"                                                        최근 종가가 3달 전의 종가보다 28% 이상 하락하고 1년 전의 종가보다 45% 이상 하락했으므로 작업을 건너뜁니다.")
                continue


        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X, Y = create_dataset(scaled_data, LOOK_BACK)

        # Python 객체 대신 TensorFlow 텐서를 사용
        # Convert the scaled_data to a TensorFlow tensor
        # scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
        # 30일 구간의 데이터셋, (365 - 30 + 1)-> 336개의 데이터셋
        # X, Y = create_dataset(scaled_data_tensor.numpy(), LOOK_BACK)  # numpy()로 변환하여 create_dataset 사용

        if len(X) < 2 or len(Y) < 2:
            print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다.")
            continue

        # 난수 데이터셋 분할
        # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
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
                  validation_data=(X_val, Y_val), callbacks=[early_stopping])

        close_scaler = MinMaxScaler()
        close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

        # 예측, 입력 X만 필요하다
        predictions = model.predict(X[-PREDICTION_PERIOD:])
        predicted_prices = close_scaler.inverse_transform(predictions).flatten()

        # 텐서 입력 사용하여 예측 실행 (권고)
        # TensorFlow가 함수를 그래프 모드로 변환하여 성능을 최적화하지만,
        # 이 과정에서 입력 데이터에 따라 미묘한 차이가 발생하거나 예상치 못한 동작을 할 수 있다

        # predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))
        # predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()

        model.save(model_file_path)

        last_close = data['종가'].iloc[-1]
        future_return = (predicted_prices[-1] / last_close - 1) * 100

        # 성장률 이상만
        if future_return < EXPECTED_GROWTH_RATE:
            continue

        # 각 종목의 튜플에 저장한다
        if ticker in ticker_returns:
            ticker_returns[ticker].append(future_return)
        else:
            ticker_returns[ticker] = [future_return]

        saved_tickers.append(ticker)

        # 그래프 생성
        extended_prices = np.concatenate((data['종가'].values, predicted_prices))
        extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
        last_price = data['종가'].iloc[-1]

        plt.figure(figsize=(16, 8))
        plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
        plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
        plt.title(f'{today_us}   {stock_name} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)

        # 디렉토리 내 파일 검색 및 삭제
        for file_name in os.listdir(output_dir):
            if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
                print(f"Deleting existing file: {file_name}")
                os.remove(os.path.join(output_dir, file_name))

        # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
        final_file_name = f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ].png'
        final_file_path = os.path.join(output_dir, final_file_name)
        # print(final_file_name)
        plt.savefig(final_file_path)
        plt.close()


    print(saved_tickers)
    # for file_name in os.listdir(output_dir):
    #     if file_name.startswith(today):
    #         print(f"{file_name}")

####################################

results = []

for ticker in saved_tickers:
    # 튜플에 5개의 데이터가 있으면 진행
    if len(ticker_returns.get(ticker, [])) == 5:
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        avg_future_return = sum(ticker_returns[ticker]) / 5
        results.append((avg_future_return, stock_name)) # 튜플

# avg_future_return을 기준으로 내림차순 정렬
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, stock_name in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} ====")
































# 주식 데이터와 기본적인 재무 데이터를 가져온다
# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     if 'PER' not in fundamental.columns:
#         print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
#         fundamental['PER'] = 0  # PER 열이 없는 경우 0으로 채움
#
#     # PER 값이 NaN인 경우 0으로 채움
#     fundamental['PER'] = fundamental['PER'].fillna(0)
#     data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
#     return data

# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     daily_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker) # 기본 일별
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     # 'PER' 컬럼이 존재하는지 먼저 확인
#     if 'PER' not in daily_fundamental.columns:
#         # 일별 데이터에서 PER 정보가 없으면 월별 데이터 요청
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PER' in monthly_fundamental.columns:
#             # 월별 PER 정보를 일별 데이터에 매핑
#             daily_fundamental['PER'] = monthly_fundamental['PER'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             # 월별 PER 정보도 없는 경우 0으로 처리
#             daily_fundamental['PER'] = 0
#     else:
#         # 일별 PER 데이터 사용, NaN 값 0으로 채우기
#         daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#
#     # PER 데이터가 없으면 0으로 채우기
#     if 'PER' not in daily_fundamental.columns or daily_fundamental['PER'].isnull().all():
#         print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
#         daily_fundamental['PER'] = 0
#
#     # PER 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#     data = pd.concat([ohlcv, daily_fundamental[['PER']]], axis=1).fillna(0)
#     return data

# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     daily_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker) # 기본 일별
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     # 'PER' 컬럼이 존재하는지 먼저 확인
#     if 'PER' not in daily_fundamental.columns:
#         # 일별 데이터에서 PER 정보가 없으면 월별 데이터 요청
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PER' in monthly_fundamental.columns:
#             # 월별 PER 정보를 일별 데이터에 매핑
#             daily_fundamental['PER'] = monthly_fundamental['PER'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             daily_fundamental['PER'] = 0
#     else:
#         daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#
#     # 'PBR' 컬럼이 존재하는지 먼저 확인
#     if 'PBR' not in daily_fundamental.columns:
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PBR' in monthly_fundamental.columns:
#             daily_fundamental['PBR'] = monthly_fundamental['PBR'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             daily_fundamental['PBR'] = 0
#     else:
#         daily_fundamental['PBR'] = daily_fundamental['PBR'].fillna(0)
#
#     # PER 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#     # PBR 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PBR'] = daily_fundamental['PBR'].fillna(0)
#
#     # 필요한 데이터만 선택하여 결합
#     data = pd.concat([ohlcv[['종가', '거래량']], daily_fundamental[['PER', 'PBR']]], axis=1).fillna(0)
#
#     return data