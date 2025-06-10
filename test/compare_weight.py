import FinanceDataReader as fdr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def invert_scale(scaled_preds, scaler, feature_index=3):
    inv_preds = []
    for row in scaled_preds:
        temp = np.zeros((len(row), scaler.n_features_in_))
        temp[:, feature_index] = row
        inv = scaler.inverse_transform(temp)[:, feature_index]
        inv_preds.append(inv)
    return np.array(inv_preds)

# 여러 종목코드 예시
tickers = ['087010', '000660', '005930']

for ticker in tickers:
    print(f'\n### {ticker} 모델 학습/예측 시작 ###')

    data = fdr.DataReader(ticker, '2024-01-01', '2025-06-09')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    forecast_horizon = 5

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length:i+sequence_length+forecast_horizon, 3])

    X = np.array(X)
    Y = np.array(y)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 모델 정의 함수로 빼면 더 깔끔
    def build_model(input_shape, forecast_horizon):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(forecast_horizon))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_model((X_train.shape[1], X_train.shape[2]), forecast_horizon)

    # 가중치 파일명: 종목별로
    weights_file = f'weights_{ticker}.h5'

    # 파일이 있으면 가중치 불러오기(이어학습), 없으면 새로 시작
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print(f"{ticker}: 이전 가중치 로드")
    else:
        print(f"{ticker}: 새 모델 학습")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, Y_train,
        epochs=200,
        batch_size=16,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop]
    )

    # 가중치 저장, 학습이 끝난 후(또는 임의 시점)에 수동으로 가중치를 저장/불러오기
    model.save_weights(weights_file)
    print(f"{ticker}: 학습 후 가중치 저장됨 -> {weights_file}")

    # 예측 및 복원
    predictions = model.predict(X_val)
    predictions_inv = invert_scale(predictions, scaler)
    Y_val_inv = invert_scale(Y_val, scaler)

    # 결과 그래프
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.title(f'{ticker} Stock Price Prediction')
    plt.plot(Y_val_inv.mean(axis=1), label='Actual Mean')
    plt.plot(predictions_inv.mean(axis=1), label='Predicted Mean')
    plt.xlabel(f'Next {forecast_horizon} days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
