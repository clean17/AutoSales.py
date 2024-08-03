from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
def predict_stock_price(data, look_back=1, days_to_predict=7):
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

# 오늘 날짜와 지난 3개월의 날짜를 문자열 형태로 설정합니다.
today = datetime.today().strftime('%Y%m%d')
three_months_ago = (datetime.today() - timedelta(days=90)).strftime('%Y%m%d')

# KOSPI 시장의 모든 종목 코드를 가져옵니다.
tickers = stock.get_market_ticker_list(market="KOSPI")

predicted_stocks = {}
count = 0  # 카운터 초기화

# 각 종목에 대해 데이터를 가져오고, 데이터가 충분히 있는 경우에만 처리를 계속합니다.
for ticker in tickers:
    count += 1  # 카운터 증가
    data = fetch_stock_data(ticker, three_months_ago, today)
    if data is not None and data.shape[0] > 50:  # 데이터가 충분한 경우에만 처리합니다.
        print(f"Processing {count}/{len(tickers)}: {ticker}")
        predictions = predict_stock_price(data)
        last_close = data['종가'].iloc[-1]
        future_return = (predictions[-1] / last_close - 1) * 100
        if future_return > 20:
            predicted_stocks[ticker] = (future_return, data['종가'], predictions)
    else:
        print(f"Insufficient data for {ticker}")  # 데이터가 충분하지 않은 경우 메시지를 출력합니다.

# 수익률 20% 이상인 종목의 그래프 시각화
for ticker, (future_return, actual_prices, predicted_prices) in predicted_stocks.items():
    last_date = actual_prices.index[-1]
    prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=8)

    plt.figure(figsize=(20, 6))
    plt.plot(actual_prices.index, actual_prices, label=f'Actual Prices ({ticker})', color='blue')
    predicted_prices_with_continuity = np.insert(predicted_prices, 0, actual_prices.iloc[-1])
    plt.plot(prediction_dates, predicted_prices_with_continuity, label=f'Predicted Prices ({ticker})', color='red', linestyle='--')

    plt.title(f'Actual Prices vs Predicted Prices for {ticker} (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
