# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import sys
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import create_multistep_dataset, fetch_stock_data, add_technical_features, create_lstm_model, \
    get_kor_ticker_dict_list, drop_trading_halt_rows

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

'''
15/3이 예측 좋다?
400~410 정도가 좋아
'''
DATA_COLLECTION_PERIOD = 400
PREDICTION_PERIOD = 3
LOOK_BACK = 15

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pickle_dir = os.path.join(ROOT_DIR, 'pickle')

# 데이터 수집
# tickers = ['358570']
tickers = ['008970', '006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']
tickers_dict = get_kor_ticker_dict_list()


# 혹시라도 1D가 되었다면 강제 2D 복원
def ensure_2d(y):
    return y.reshape(-1, 1) if y.ndim == 1 else y



for i in range(1):
    # period = DATA_COLLECTION_PERIOD + (10*i)
    # print('period : ', period)
    start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
    # start_date = (datetime.today() - timedelta(days=period)).strftime('%Y%m%d')
    # print('LOOK_BACK : ', (LOOK_BACK + i))

    total_mse = 0
    total_r2 = 0
    total_cnt = 0

    for count, ticker in enumerate(tickers):
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

        # 1. 데이터 수집
        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        data = pd.read_pickle(filepath)

        # 1-1. 우선 거래정지/이상치 행 제거
        data, removed_idx = drop_trading_halt_rows(data)
        # if len(removed_idx) > 0:
        #     print(f"거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")

        # 2. 2차 생성 feature
        data = add_technical_features(data)

        # 3. 결측 제거
        threshold = 0.1  # 10%
        cols_to_drop = [
            col for col in data.columns
            if (data[col].isna().mean() > threshold) or (np.isinf(data[col]).mean() > threshold)
        ]
        if len(cols_to_drop) > 0:
            data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            print("Drop candidates:", cols_to_drop)
            # continue


        # ---- 전처리: NaN/inf 제거 및 피처 선택 ----
        feature_cols = [
            '시가', '고가', '저가', '종가', 'Vol_logdiff',
            'RSI14',

            # 'MA5_slope',
            # 'CCI14',
            # 'STD20', 'UpperBand', 'LowerBand',
            # 'UltimateOsc',
        ]

        # 4. 피쳐, 무한대 필터링
        cols = [c for c in feature_cols if c in data.columns]  # 순서 보존
        df = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
        X_df = df.dropna()
        idx_close = feature_cols.index('종가')

        # 5. 스케일링, 시점 마스크
        split = int(len(X_df) * 0.8)
        scaler_X = StandardScaler().fit(X_df.iloc[:split])  # 원시 train 구간만, 중복 윈도우 때문에 같은 시점 행이 여러 번 들어가는 왜곡 방지
        X_all_2d = scaler_X.transform(X_df)                 # 전체 변환 (누수 없음)

        LOOK_BACK = 15
        N_FUTURE = 3

        # 이미 스케일된 데이터셋
        X_all, Y_all, t0 = create_multistep_dataset(X_all_2d, LOOK_BACK, N_FUTURE, idx=idx_close, return_t0=True)

        train_mask = (t0 + N_FUTURE - 1) < split
        val_mask   = (t0 >= split)

        X_train, Y_train = X_all[train_mask], Y_all[train_mask]
        X_val,   Y_val   = X_all[val_mask],   Y_all[val_mask]

        # ---- Train/Test split (스케일러는 Train으로만 fit) ----
        Y_train = ensure_2d(Y_train)
        Y_val  = ensure_2d(Y_val)

        # ---- y 스케일링: Train으로만 fit ---- (타깃이 수익률이면 생략 가능)
        scaler_y = StandardScaler().fit(Y_train)
        y_train_scaled = scaler_y.transform(Y_train)
        y_val_scaled  = scaler_y.transform(Y_val)

        # dtype & 메모리 연속성(권장: float32, C-order)
        X_train = np.asarray(X_train, dtype=np.float32, order="C")
        X_val   = np.asarray(X_val,   dtype=np.float32, order="C")
        y_train_scaled = np.asarray(y_train_scaled, dtype=np.float32, order="C")
        y_val_scaled   = np.asarray(y_val_scaled,   dtype=np.float32, order="C")

        if X_train.shape[0] < 50:
            print('샘플 부족 : ', X_train.shape)
            continue

        model = create_lstm_model((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD,
                                      lstm_units=[32], dense_units=[16])

        # 콜백 설정
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
        model.fit(
            X_train, y_train_scaled,
            batch_size=8, epochs=200, verbose=0, shuffle=False,
            validation_data=(X_val, y_val_scaled),
            callbacks=[early_stop]
        )

        y_pred_scaled = model.predict(X_val, batch_size=256, verbose=0)

        # -------- 평가 ---------
        y_true = scaler_y.inverse_transform(y_val_scaled)     # (N, H)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)    # (N, H)

        last_close = X_val[:, -1, idx_close]
        y_naive = np.repeat(last_close[:, None], y_true.shape[1], axis=1)


        # --- Naive: persistence (가격용) ---
        # X_val: (N,T,F), idx_close: 종가 열
        last_close = X_val[:, -1, idx_close]    # 반드시 '원 단위' 원본으로 맞추세요
        y_naive = np.repeat(last_close[:, None], y_true.shape[1], axis=1)


        # --- 멀티스텝 집계 ---
        def per_horizon_metrics(y_true, y_pred):
            H = y_true.shape[1]
            out = []
            for h in range(H):
                out.append({
                    "h": h+1,
                    "RMSE": rmse(y_true[:,h], y_pred[:,h]),
                    "MAE" : mae (y_true[:,h], y_pred[:,h]),
                    "R2"  : r2_score(y_true[:,h], y_pred[:,h])
                })
            return out

        # Model vs Naive
        print("== Aggregate ==")
        print("RMSE(model) :", rmse(y_true, y_pred))
        print("RMSE(naive) :", rmse(y_true, y_naive))
        print("ratio       :", rmse(y_true, y_pred) / rmse(y_true, y_naive))
        print("SMAPE(model):", smape(y_true, y_pred))

        print("\n== Per-horizon ==")
        for row in per_horizon_metrics(y_true, y_pred):
            h = row["h"]
            r_model = rmse(y_true[:,h-1], y_pred[:,h-1])
            r_naive = rmse(y_true[:,h-1], y_naive[:,h-1])
            print(f"h={h:>2}  RMSE(model)={r_model:.2f}  RMSE(naive)={r_naive:.2f}  ratio={r_model/r_naive:.3f}")

        # 방향 정확도(hit rate) - 방향을 맞췄는지 비율 // 무작위 기준: 0.5, 일반 관찰치: 0.4~0.6
        def hit_rate(y_true, y_pred):
            # 마지막 스텝 기준 (원하면 모든 스텝 평균도 가능)
            y_t = y_true[:,-1] - y_true[:,-2] if y_true.shape[1] > 1 else y_true[:,-1]
            p_t = y_pred[:,-1] - y_true[:,-2] if y_true.shape[1] > 1 else y_pred[:,-1]
            return np.mean((y_t >= 0) == (p_t >= 0))

        print("Hit rate:", hit_rate(y_true, y_pred))

        # MSE
        mse = mean_squared_error(Y_val, y_pred_scaled)
        # print("MSE:", mse)
        total_mse += mse

        # R-squared; (0=엉망, 1=완벽)
        r2 = r2_score(Y_val, y_pred_scaled)
        # print("R-2:", r2)
        total_r2 += r2
        total_cnt += 1
        print('')


    print('mse : ', total_mse/total_cnt)
    print('r2 : ', total_r2/total_cnt)






'''
훈련용/검증용 loss

0.01~0.02 이하:
  → 매우 우수 (1~2% 미만 평균 오차)
0.02~0.05:
  → 실전 충분, 대부분의 실전 예측에서 이 정도면 OK
0.05~0.08:
  → 크다고 느껴질 수 있음, 신호가 “정확히 맞는지” 확인 필요
0.1 이상:
  → 오버피팅/과소적합/샘플 부족 가능성.
  실전에서는 예측 신호 품질 직접 확인 필요
  
0.01 이하면 매우 잘 맞추는 모델(스케일이 0~1로 정규화되어 있으면)
0.005 이하면 “상당히 우수”
0.001 이하면 “거의 오버핏/탁월”

R2
0.7~0.8: 실전에서 "꽤 우수한" 성능 (시계열/주가 등 변동성 큰 데이터 기준)
0.8~0.9: "아주 잘 맞추는" 예측
0.9 이상: 거의 완벽 (실전에서는 드물며, 오버피팅 의심도)
0.5~0.7: “적당히 쓸만한” 모델
0.5 이하: 실전 활용도 낮음 (정확도 개선 필요)
'''



"""
combined_loss = list(history2.history['loss'])
combined_val_loss = list(history2.history['val_loss'])

# loss 그래프 그리기
# 0에 수렴하면 학습이 정상, 다시 오르면 과적합
plt.figure(figsize=(10, 5))
plt.plot(combined_loss, label='Train Loss')
plt.plot(combined_val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
"""
