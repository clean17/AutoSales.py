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
LOOK_BACK = 18
DATA_COLLECTION_PERIOD = 300

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

tickers = ['077970', '079160', '112610', '025540', '003530', '357880', '131970', '009450', '310210', '353200', '136150', '064350', '066575', '005880', '272290', '204270', '066570', '456040', '373220', '096770', '005490', '006650',
           '042700', '068240', '003280', '067160', '397030', '480370', '085660', '328130', '476040', '241710', '357780', '232140', '011170', '020180', '074600', '042000', '003350', '065350', '004490', '482630', '005420', '033100',
           '018880', '417200', '332570', '058970', '011790', '053800', '338220', '195870', '010950', '455900', '082740', '225570', '445090', '068760', '007070', '361610', '443060', '089850', '413640', '005850', '141080', '005380',
           '098460', '277810', '011780', '005810', '075580', '112040', '012510', '240810', '403870', '376900', '001740', '035420', '103140', '068270', '013990', '001450', '457190', '293580', '475150', '280360', '097950', '058820']

tickers = ['077970', '079160', '112610', '025540', '003530', '357880', '131970', '009450', '310210', '353200', '136150', '064350', '066575', '005880', '272290', '204270', '066570', '456040', '373220', '096770', '005490', '006650']




all_importances1 = []
all_importances2 = []
all_importances3 = []

for ticker in tickers:
    data = add_technical_features(fetch_stock_data(ticker, start_date, today))

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # day_range_pct_t, 거래량, 종가, rsi, 고가, m5s, pbr, 저가, m5 m10
    feature_cols = [
        '종가', '고가', '저가', 'PBR', 'MA5', 'MA10', 'MA5_slope',
        '거래량', 'RSI14', 'day_range_pct',
    ]
    flattened_feature_names = []
    for t in range(LOOK_BACK):
        for f in feature_cols:
            flattened_feature_names.append(f"{f}_t{-LOOK_BACK + t + 1}")

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
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
    all_importances1.append(rf.feature_importances_)

    # 2차 필터링
    result = permutation_importance(rf, X_val, y_val, n_repeats=10) # 각 feature를 10번씩 셔플해서 평균냄
    all_importances2.append(result.importances_mean)  # 중요도 평균값만 누적

    # 3차 필터링
    selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=30)
    selector.fit(X_train, y_train)
    all_importances3.append(selector.support_)



# (종목 수, feature 수) -> feature별 평균
all_importances1 = np.array(all_importances1)
mean_importances1 = all_importances1.mean(axis=0)

all_importances2 = np.array(all_importances2)
mean_importances2 = all_importances2.mean(axis=0)

all_importances3 = np.array(all_importances3)
mean_importances3 = all_importances3.mean(axis=0)

# 중요도 높은 순으로 상위 100개 출력 (feature 개수에 맞게 조정)
for i in np.argsort(mean_importances1)[::-1][:20]:
    print(f"{flattened_feature_names[i]}: {mean_importances1[i]:.4f}")

print(' ')
print(' ---------------------------------------- ')
print(' ')

for i in np.argsort(mean_importances2)[::-1][:20]:
    print(f"{flattened_feature_names[i]}: {mean_importances2[i]:.4f}")

print(' ')
print(' ---------------------------------------- ')
print(' ')

for i in np.argsort(mean_importances3)[::-1][:100]:
    print(f"{flattened_feature_names[i]}: {mean_importances3[i]:.4f}")

'''
# 1차 0.01 이상 컷오프 #
종가_t0: 0.3184
고가_t0: 0.1705
PBR_t0: 0.1050
저가_t0: 0.0949
MA10_t0: 0.0266
고가_t-1: 0.0116
저가_t-1: 0.0115
PBR_t-1: 0.0109
MA5_t0: 0.0107
종가_t-1: 0.0104

종가, 고가, pbr, 저가, m10, m5


# 2차 0.02 이상 컷오프 #
day_range_pct_t-11: 0.3163
거래량_t-2: 0.2855
종가_t0: 0.1755
day_range_pct_t-12: 0.1460
day_range_pct_t-5: 0.1119
RSI14_t-7: 0.0947
고가_t-17: 0.0882
거래량_t-11: 0.0843
종가_t-7: 0.0753
MA5_slope_t-16: 0.0717
PBR_t-1: 0.0697
저가_t-7: 0.0462
MA5_t0: 0.0383
MA10_t0: 0.0233
거래량_t-7: 0.0213

day_range_pct_t, 거래량, 종가, rsi, 고가, m5s, pbr, 저가, m5 m10


# 3차 #
종가_t0: 1.0000
저가_t0: 0.9545
고가_t0: 0.9545
PBR_t0: 0.8636
MA5_t0: 0.5909
저가_t-1: 0.5909
BB_perc_t0: 0.5000
종가_t-1: 0.5000
거래량_t-8: 0.4091
고가_t-1: 0.4091
MA10_t-17: 0.3636
MA10_t0: 0.3636
거래량_t-17: 0.3182
MA10_t-16: 0.3182
거래량_t-11: 0.3182
RSI14_t-9: 0.3182
MA5_slope_t-1: 0.3182
BB_perc_t-8: 0.3182
MA5_slope_t0: 0.3182

종가_t0, 저가_t0, 고가_t0, PBR_t0, MA5_t0, BB_perc_t0



실제 "예측력"에 더 가까운 평가는 permutation_importance 쪽!
"진짜 이 feature가 없으면 모델이 예측을 못하냐?"를 기준으로 보기 때문

RFE는 변수 수를 줄이거나 자동화할 때 참고용
불필요한 feature 자동 제거, 대략적인 feature subset 선정엔 유용

하지만, 실제 성능 향상/실전 해석에는 permutation_importance가 더 직관적
'''
