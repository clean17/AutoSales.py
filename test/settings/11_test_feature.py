import matplotlib
import os
import sys
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from send2trash import send2trash
import ast
import numpy as np
from sklearn.model_selection import train_test_split

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data, compute_rsi

PREDICTION_PERIOD = 4
LOOK_BACK = 20 # validation loss 값 테스트 필요
DATA_COLLECTION_PERIOD = 300
window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

ticker = '103140'


data = fetch_stock_data(ticker, start_date, today)
print(len(data))


data['MA5'] = data['종가'].rolling(window=5).mean()
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
# 캔들패턴 (양봉/음봉, 장대양봉 등)
# data['is_bullish'] = (data['종가'] > data['시가']).astype(int) # 양봉이면 1, 음봉이면 0
# 장대양봉(시가보다 종가가 2% 이상 상승)
# data['long_bullish'] = ((data['종가'] - data['시가']) / data['시가'] > 0.02).astype(int)
# 당일 변동폭 (고가-저가 비율)
# data['day_range_pct'] = (data['고가'] - data['저가']) / data['저가']


# NaN 개수
# print("NaN 개수")
# print(data.isna().sum())

# inf/-inf 개수 (각 컬럼별)
# print("\ninf/-inf 개수")
# print(np.isinf(data).sum())

total = len(data)
print("NaN 비율(%)")
print((data.isna().sum() / total * 100).round(2))

# print("\ninf/-inf 비율(%)")
# print((np.isinf(data).sum() / total * 100).round(2))
'''
일반적으로 10% 이상 NaN/inf면 문제
만약 “NaN이 1~2개밖에 없음” → 그냥 0이나 평균값으로 대체해도 무방
“절반이 NaN/inf다” → 그 feature는 제거하는 것이 안전
'''
#  NaN/inf를 자동 제거하려면
# NaN/inf가 전체의 10% 이상인 feature만 자동 drop
threshold = 0.1  # 10%
cols_to_drop = [
    col for col in data.columns
    if (data[col].isna().mean() > threshold) or (np.isinf(data[col]).mean() > threshold)
]
print("Drop candidates:", cols_to_drop)

# 데이터셋 생성
scaler = MinMaxScaler(feature_range=(0, 1))
# feature_cols = ['시가', '고가', '저가', '종가', '거래량', 'MA20', 'UpperBand', 'LowerBand', 'PER', 'PBR']
feature_cols = [
    '종가', 'MA15_slope', 'RSI14', 'BB_perc', 'Volume_change'
]

flattened_feature_names = []
for t in range(LOOK_BACK):
    for f in feature_cols:
        flattened_feature_names.append(f"{f}_t{-LOOK_BACK + t + 1}")

X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
# print(np.isfinite(X_for_model).all())  # True면 정상, False면 비정상
# print(np.where(~np.isfinite(X_for_model)))  # 문제 있는 위치 확인
scaled_data = scaler.fit_transform(X_for_model)
X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD)



'''
랜덤포레스트/트리 계열

RandomForestRegressor 등으로
전체 feature를 넣고, feature 중요도(feature_importances_; 트리 기반 내장 중요도)를 평가
각 feature를 얼마나 자주, 얼마나 크게 사용했는지를 기반으로 계산

>> 트리가 내부적으로 중요하다고 생각한 split
빠름, 계산비용 적음, 순전히 트리 내부 기준(모델 학습 과정에만 근거)
feature가 너무 많을 때 빠르게 1차 필터링
'''
from sklearn.ensemble import RandomForestRegressor

n_samples, look_back, n_features = X.shape
X_rf = X.reshape(n_samples, look_back * n_features)
Y_rf = Y[:, 0]   # 또는 Y_rf = Y.reshape(-1)

X_train, X_val, y_train, y_val = train_test_split(
    X_rf, Y_rf, test_size=0.2, shuffle=False
)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)  # X_train: (샘플, feature), y_train: (샘플, )
# importances = rf.feature_importances_

# # 중요도 순으로 정렬
# for i in np.argsort(importances)[::-1]:
#     print(f"{flattened_feature_names[i]}: {importances[i]:.4f}")



'''
Permutation Importance (모델 agnostic, 대부분 모델에 가능) (셔플 중요도, 모델-불문 실제 평가)

모델의 예측이 특정 feature에 얼마나 의존하는지
→ feature의 값만 무작위로 섞어(셔플해서)
모델 성능이 얼마나 떨어지는지를 관찰 (실험적 검증)

X_val, y_val 등 “학습에 쓰지 않은 데이터”에서 평가
(실제 성능 하락폭으로 평가 → 일반화 성능 기준)

feature 간 상호작용까지 반영

실전에서 더 신뢰받는 평가
(특히, feature가 서로 의존적일 때)

>> 실전에서 이 feature가 망가지면(섞이면) 진짜 성능이 떨어지는가?
실전(운영)에서 진짜 영향력 있는 feature가 궁금할 때

n_repeats=10: 각 feature를 10번씩 셔플해서 평균냄
'''
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_val, y_val, n_repeats=10)
for i in result.importances_mean.argsort()[::-1]: # 가장 중요한 feature부터 내림차순 정렬
    print(f"{flattened_feature_names[i]}: {result.importances_mean[i]:.4f}")



'''
반복적 특징 제거 (Recursive Feature Elimination, RFE)
자동으로 덜 중요한 feature를 제거해서,
중요한 feature만 남기는 대표적인 방법

n_features_to_select: 최종적으로 남길 feature 개수 

RFE는 feature가 수십~수백개일 때 특히 효과적
'''
from sklearn.feature_selection import RFE

selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=20)
selector = selector.fit(X_train, y_train)
selected_features = np.array(flattened_feature_names)[selector.support_]
print(selected_features)