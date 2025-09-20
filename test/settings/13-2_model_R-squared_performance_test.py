import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import sys
import pickle
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, get_kor_ticker_dict_list, \
    drop_trading_halt_rows, fetch_stock_data

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

'''
15/3이 예측 좋다?
400~410 정도가 좋아
'''

def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

DATA_COLLECTION_PERIOD = 400
PREDICTION_PERIOD = 3
LOOK_BACK = 15


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pickle_dir = os.path.join(ROOT_DIR, 'pickle')
# pickle_dir = os.path.join(ROOT_DIR, 'pickle_us')


# 데이터 수집
# tickers = ['006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']
# tickers = ['MNKD', 'ESPR', 'ALKS', 'LASR', 'TLRY', 'TSLA', 'SNDL', 'INSG', 'SABR', 'TBPH', 'VFF', 'AVDL', 'EVLV']
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())

today = datetime.today().strftime('%Y%m%d')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')

for i in range(1):

    total_mse = 0
    total_r2 = 0
    total_cnt = 0

    for count, ticker in enumerate(tickers):
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

        # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)
            data = fetch_stock_data(ticker, start_five_date, today)
        else:
            df = pd.DataFrame()
            data = fetch_stock_data(ticker, start_date, today)

        # 중복 제거 & 새로운 날짜만 추가
        if not df.empty:
            # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
            new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
            df = pd.concat([df, new_rows])
        else:
            df = data

        # 파일 저장
        df.to_pickle(filepath)
        data = df


        # 0) 우선 거래정지/이상치 행 제거
        data, removed_idx = drop_trading_halt_rows(data)
        if len(removed_idx) > 0:
            print(f"거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")

        data = add_technical_features(data)


        threshold = 0.1  # 10%
        # isna() : pandas의 결측값(NA) 체크. NaN, None, NaT에 대해 True
        # mean() : 평균
        # isinf() : 무한대 체크
        cols_to_drop = [
            col for col in data.columns
            if (data[col].isna().mean() > threshold) or (np.isinf(data[col]).mean() > threshold)
        ]
        if len(cols_to_drop) > 0:
            # inplace=True : 반환 없이 입력을 그대로 수정
            # errors='ignore' : 목록에 없는 칼럼 지우면 에러지만 무시
            data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            print("Drop candidates:", cols_to_drop)


        # 한국/영문 칼럼 자동 식별
        col_o = _col(df, '시가',   'Open')
        col_h = _col(df, '고가',   'High')
        col_l = _col(df, '저가',   'Low')
        col_c = _col(df, '종가',   'Close')
        col_v = _col(df, '거래량', 'Volume')


        # 학습에 쓸 피처
        feature_cols = [
            col_o, col_l, col_h, col_c,
            'Vol_logdiff',
            'ma10_gap',
            'MA5_slope',
        ]
        # col_o, col_l, col_h, col_c, Vol_logdiff, ma10_gap, MA5_slope
        # 0) NaN/inf 정리
        data = data.replace([np.inf, -np.inf], np.nan)

        # 1) feature_cols만 남기고, 그 안에서만 dropna
        feature_cols = [c for c in feature_cols if c in data.columns]
        X_df = data.dropna(subset=feature_cols).loc[:, feature_cols]

        # (선택) 상수열 제거: 스케일링/학습 안정화
        const_cols = [c for c in X_df.columns if X_df[c].nunique() <= 1]
        if const_cols:
            X_df = X_df.drop(columns=const_cols)
            feature_cols = [c for c in feature_cols if c not in const_cols]

        # 2) 시계열 분리 후, train으로만 fit → val은 transform만
        split = int(len(X_df) * 0.9)
        scaler = MinMaxScaler(feature_range=(0,1))
        # fit: 스케일러가 데이터의 통계값을 학습         >> 훈련 세트: 훈련 데이터로만 통계(평균·최대/최소 등)를 학습 + 변환
        # transform: 이미 학습된 통계값을 써서 값을 변환 >> 검증/테스트 세트: 훈련에서 배운 통계로만 변환
        X_tr_2d = scaler.fit_transform(X_df.iloc[:split].values)
        X_va_2d = scaler.transform(X_df.iloc[split:].values)

        # 3) 윈도잉 (블록별로 따로 만들어 경계 혼합 방지)
        idx_close = feature_cols.index(col_c)
        # print('idx_close',idx_close)

        X_train, Y_train = create_multistep_dataset(X_tr_2d, LOOK_BACK, PREDICTION_PERIOD, idx=idx_close)
        X_val,   Y_val   = create_multistep_dataset(X_va_2d, LOOK_BACK, PREDICTION_PERIOD, idx=idx_close)

        # 4) 최종 안전 체크
        import numpy as np
        for name, arr in [('X_train',X_train),('Y_train',Y_train),('X_val',X_val),('Y_val',Y_val)]:
            assert np.isfinite(arr).all(), f"{name} has NaN/inf: check preprocessing"

        # 실수 주의
        # scaler.fit_transform(X_all) 후에 train/val 나누기 → ❌ 누수
        # transform을 fit 전에 호출 → ❌ 에러 발생(아직 통계가 없음)
        # X_for_model = data.dropna(subset=feature_cols) # 결측 제거; feature_col에 들어있는 칼럼 중 하나라도 NaN인 row를 삭제
        # scaled_data = scaler.fit_transform(X_for_model)



        if X_train.shape[0] < 50:
            print('샘플 부족 : ', X_train.shape)
            continue


        model = create_lstm_model((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD,
                                  lstm_units=[128,64], dense_units=[64,32])

        # 콜백 설정
        from tensorflow.keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원
        history = model.fit(X_train, Y_train, batch_size=8, epochs=200,
                            validation_data=(X_val, Y_val),
                            shuffle=False, verbose=0, callbacks=[early_stop])


        # ====== 평가 ======
        # X_val : 검증에 쓸 여러 시계열 구간의 집합
        # predictions : 검증셋 각 구간(윈도우)에 대해 미래 PREDICTION_PERIOD만큼의 예측치 반환
        # shape: (검증샘플수, 예측일수)
        preds = model.predict(X_val, verbose=0)
        # print('predictions (샘플, 예측일)', predictions.shape)

        # MSE
        mse = mean_squared_error(Y_val, preds)
        total_mse += mse

        # R-squared; (0=엉망, 1=완벽)
        r2 = r2_score(Y_val, preds)
        total_r2 += r2
        total_cnt += 1


    print('MSE_avg : ', total_mse/total_cnt)
    print('R-squared_avg : ', total_r2/total_cnt)






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
