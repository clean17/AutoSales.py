from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


# 예측 기간 (1달)
prediction_period = 30
# 예측 성장률 (20%)
expected_growth_rate = 20
# 데이터 수집 기간 (6개월)
data_collection_period = 180


def fetch_stock_data(ticker, fromdate, todate):
    # 주식 데이터와 기본적인 재무 데이터를 가져옵니다.
    ohlcv = stock.get_market_ohlcv_by_date(fromdate=fromdate, todate=todate, ticker=ticker)
    fundamental = stock.get_market_fundamental_by_date(fromdate=fromdate, todate=todate, ticker=ticker)
    
    # PER 데이터가 있는지 확인하고 없으면 None을 반환합니다.
    if 'PER' not in fundamental.columns:
        print(f"PER data not available for {ticker}")
        return None
    # 주가 데이터와 PER 데이터를 합쳐서 반환합니다.
    return pd.concat([ohlcv['종가'], fundamental['PER']], axis=1).dropna()

# LSTM 모델 학습 및 예측 함수 정의
def predict_stock_price(data, look_back=1, days_to_predict=prediction_period):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, look_back)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, batch_size=32, epochs=20, verbose=0)

    predictions = model.predict(X[-days_to_predict:])
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((days_to_predict, 1))), axis=1))[:, 0]

    return predictions


# 오늘 날짜와 지난 x개월의 날짜를 문자열 형태로 설정합니다.
today = datetime.today().strftime('%Y%m%d')
three_months_ago = (datetime.today() - timedelta(days=data_collection_period)).strftime('%Y%m%d')

# KOSPI 시장의 모든 종목 코드를 가져옵니다.
tickers = stock.get_market_ticker_list(market="KOSPI")

# 종목 코드와 이름 딕셔너리 생성
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

predicted_stocks = {}
count = 0  # 카운터 초기화

# 각 종목에 대해 데이터를 가져오고, 데이터가 충분히 있는 경우에만 처리를 계속합니다.
for ticker in tickers:
# for ticker in tickers[:1]: # test
    count += 1  # 카운터 증가
    data = fetch_stock_data(ticker, three_months_ago, today)
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')

    if data is not None and data.shape[0] > 50:  # 데이터가 충분한 경우에만 처리합니다.
        print(f"Processing {count}/{len(tickers)}: {stock_name} {ticker}")
        predictions = predict_stock_price(data)
        last_close = data['종가'].iloc[-1]
        future_return = (predictions[-1] / last_close - 1) * 100
        if future_return > expected_growth_rate:
            predicted_stocks[ticker] = (future_return, data['종가'], predictions)
    else:
        print(f"Insufficient data for {ticker}")  # 데이터가 충분하지 않은 경우 메시지를 출력합니다.


# 이미지 저장 경로 설정 (절대경로.. C:\)
# output_dir = '/images'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# 현재 작업 디렉토리를 기준으로 output_dir 설정
output_dir = os.path.join(os.getcwd(), 'images')

# 디렉터리가 존재하지 않으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 수익률 20% 이상인 종목의 그래프 시각화
for ticker, (future_return, actual_prices, predicted_prices) in predicted_stocks.items():
    last_date = actual_prices.index[-1]
    prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_period+1)

    plt.figure(figsize=(20, 6))
    plt.plot(actual_prices.index, actual_prices, label=f'Actual Prices ({stock_name} {ticker})', color='blue')
    predicted_prices_with_continuity = np.insert(predicted_prices, 0, actual_prices.iloc[-1])
    plt.plot(prediction_dates, predicted_prices_with_continuity, label=f'Predicted Prices ({ticker})', color='red', linestyle='--')

    plt.title(f'Actual Prices vs Predicted Prices for {stock_name} {ticker} (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    # plt.show()

    last_price = actual_prices.iloc[-1]

    # 이미지 파일로 저장
    file_path = os.path.join(output_dir, f'{ticker}_{stock_name}_{last_price}.png')
    plt.savefig(file_path)
    plt.close()