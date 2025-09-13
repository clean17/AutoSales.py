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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data, add_technical_features

PREDICTION_PERIOD = 3
LOOK_BACK = 10
DATA_COLLECTION_PERIOD = 400

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

tickers = ['006490', '042670', '023160', '006800', '323410', '009540', '058970', '034020', '079550', '358570', '000155', '035720', '00680K', '035420', '012510']


all_importances1 = []
all_importances2 = []
all_importances3 = []

for ticker in tickers:
    data = add_technical_features(fetch_stock_data(ticker, start_date, today))

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = [
        '종가', '고가', 'PBR', '저가', '거래량', 'RSI14', 'ma10_gap',
    ]
    flattened_feature_names = []
    for t in range(LOOK_BACK):
        for f in feature_cols:
            flattened_feature_names.append(f"{f}_t{-LOOK_BACK + t + 1}")

    X_for_model = data.dropna(subset=feature_cols) # 결측 제거
    scaled_data = scaler.fit_transform(X_for_model)
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD, 0)


    n_samples, look_back, n_features = X.shape # 3차원의 값을 분리 할당
    X_rf = X.reshape(n_samples, look_back * n_features)
    Y_rf = Y[:, 0]   # 또는 Y_rf = Y.reshape(-1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_rf, Y_rf, test_size=0.1, shuffle=False
    )

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    # 1차 필터링
    # all_importances1.append(rf.feature_importances_)

    # 2차 필터링
    # n_repeats=10~30: 일반적, 실험/개발/EDA 단계에서 충분
    # n_repeats=50~100 이상: 중요한 보고서, 분석, 논문 등 “통계적 신뢰성”이 매우 필요할 때
    result = permutation_importance(rf, X_val, y_val, n_repeats=30) # 각 feature를 10번씩 셔플해서 평균냄
    all_importances2.append(result.importances_mean)  # 중요도 평균값만 누적

    # 3차 필터링
    # selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=30)
    # selector.fit(X_train, y_train)
    # all_importances3.append(selector.support_)



# (종목 수, feature 수) -> feature별 평균

# 중요도 높은 순으로 상위 100개 출력 (feature 개수에 맞게 조정)
# all_importances1 = np.array(all_importances1)
# mean_importances1 = all_importances1.mean(axis=0)
# for i in np.argsort(mean_importances1)[::-1][:20]:
#     print(f"{flattened_feature_names[i]}: {mean_importances1[i]:.4f}")

print(' ')
print(' ---------------------------------------- ')
print(' ')

all_importances2 = np.array(all_importances2)
mean_importances2 = all_importances2.mean(axis=0)
for i in np.argsort(mean_importances2)[::-1][:20]:
    print(f"{flattened_feature_names[i]}: {mean_importances2[i]:.4f}")

print(' ')
print(' ---------------------------------------- ')
print(' ')

# all_importances3 = np.array(all_importances3)
# mean_importances3 = all_importances3.mean(axis=0)
# for i in np.argsort(mean_importances3)[::-1][:100]:
#     print(f"{flattened_feature_names[i]}: {mean_importances3[i]:.4f}")

'''
# 1차 0.01 이상 컷오프 #

# 2차 0.02 이상 컷오프 #

# 3차 #


실제 "예측력"에 더 가까운 평가는 permutation_importance 쪽!
"진짜 이 feature가 없으면 모델이 예측을 못하냐?"를 기준으로 보기 때문

RFE는 변수 수를 줄이거나 자동화할 때 참고용
불필요한 feature 자동 제거, 대략적인 feature subset 선정엔 유용

하지만, 실제 성능 향상/실전 해석에는 permutation_importance가 더 직관적
'''
