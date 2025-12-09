import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import os, sys
from pathlib import Path

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
    else:
        raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import create_model, create_multistep_dataset, fetch_stock_data


window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수
n_future = 3
look_back = 15

# 데이터 수집
ticker = "000660"

DATA_COLLECTION_PERIOD = 400 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
today = datetime.today().strftime('%Y%m%d')

# 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
if os.path.exists(filepath):
    df = pd.read_pickle(filepath)
    data = fetch_stock_data(ticker, start_five_date, today)
else:
    df = pd.DataFrame()
    data = fetch_stock_data(ticker, start_date, today)


# 볼린저밴드
data['MA20'] = data['종가'].rolling(window=window).mean()
data['STD20'] = data['종가'].rolling(window=window).std()
data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)
# 모델 입력(feature) 만들 땐 "NaN 없는 행"만 추출해서 사용
feature_cols = ['시가', '고가', '저가', '종가', '거래량', 'MA20', 'UpperBand', 'LowerBand', 'PER', 'PBR']
X_for_model = data.dropna(subset=feature_cols) # 결측 제거
scaled_data = scaler.fit_transform(X_for_model)


X, Y = create_multistep_dataset(scaled_data, look_back, n_future)
model = create_model((X.shape[1], X.shape[2]), n_future)

# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])


# 종가 scaler fit (실제 데이터로)
close_scaler = MinMaxScaler()
close_prices = data['종가'].values.reshape(-1, 1)
close_scaler.fit(close_prices)

# 예측
X_input = scaled_data[-look_back:].reshape(1, look_back, scaled_data.shape[1])
future_preds = model.predict(X_input, verbose=0).flatten()
future_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten()

# 날짜 처리
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_future, freq='B') # 예측할 5 영업일 날짜
actual_prices = data['종가'].values # 최근 종가 배열

plt.figure(figsize=(12,6))
plt.plot(data.index, actual_prices, label='실제 가격') # 과거; data.index 받아온 날짜 인덱스
plt.plot(future_dates, future_prices, label='예측 가격', linestyle='--', marker='o', color='orange') # 예측

if all(x in data.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
    plt.plot(data.index, data['MA20'], label='20일 이동평균선') # MA20
    plt.plot(data.index, data['UpperBand'], label='볼린저밴드 상한선', linestyle='--') # Upper Band (2σ)
    plt.plot(data.index, data['LowerBand'], label='볼린저밴드 하한선', linestyle='--') # Lower Band (2σ)
    plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)

# 마지막 실제값과 첫 번째 예측값을 점선으로 연결
plt.plot(
    [data.index[-1], future_dates[0]],  # x축: 마지막 실제날짜와 첫 예측날짜
    [actual_prices[-1], future_prices[0]],  # y축: 마지막 실제종가와 첫 예측종가
    linestyle='dashed', color='gray', linewidth=1.5
)

plt.title(f'Predicted Prices for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()