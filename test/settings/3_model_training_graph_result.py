# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import matplotlib
# matplotlib.use("Agg") # plt.show() 에서 사용안함
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
pickle_dir = os.path.join(BASE_DIR, 'pickle')

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, get_kor_ticker_dict_list, \
    drop_trading_halt_rows, fetch_stock_data



# 1. 데이터 수집
ticker = '000660' # 하이닉스
# ticker = '042670'
# ticker = '006490'
# ticker = '006800'

LOOK_BACK = 15
N_FUTURE = 3



filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
data = pd.read_pickle(filepath)

# 1-1. 우선 거래정지/이상치 행 제거
data, removed_idx = drop_trading_halt_rows(data)
if len(removed_idx) > 0:
    print(f"거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")

# 2. 2차 생성 feature
data = add_technical_features(data)
# print(data.columns) # 칼럼 헤더 전체 출력

# 3. 결측 제거
threshold = 0.1  # 10%
# isna() : pandas의 결측값(NA) 체크. NaN, None, NaT에 대해 True
# mean() : 평균
# isinf() : 무한대 체크
cols_to_drop = [ # 결측치가 10% 이상인 칼럼
    col
    for col in data.columns
    if (~np.isfinite(pd.to_numeric(data[col], errors='coerce'))).mean() > threshold
]
if len(cols_to_drop) > 0:
    # inplace=True : 반환 없이 입력을 그대로 수정
    # errors='ignore' : 목록에 없는 칼럼 지우면 에러지만 무시
    data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print("Drop candidates:", cols_to_drop)

# 혹시라도 1D가 되었다면 강제 2D 복원
def ensure_2d(y):
    return y.reshape(-1, 1) if y.ndim == 1 else y

# 스케일된 ‘종가’만 원 단위로 역변환하는 도우미
def inverse_close_matrix(Y_xscale: np.ndarray,
                         scaler_X,
                         n_features: int,
                         idx_close: int) -> np.ndarray:
    """
    Y_xscale: (N, H)  # X-스케일의 '종가' 값들
    scaler_X: X에 fit했던 StandardScaler
    n_features: X의 피처 개수 (X_train.shape[2])
    idx_close: 종가 컬럼 인덱스 (X의 피처 순서 기준)
    반환: (N, H) 원 단위(가격)
    """
    N, H = Y_xscale.shape
    Z = np.zeros((N*H, n_features), dtype=float)
    Z[:, idx_close] = Y_xscale.reshape(-1)
    raw = scaler_X.inverse_transform(Z)[:, idx_close]
    return raw.reshape(N, H)

# 25.10.05
"""
'시가', '고가', '저가', '종가', 'Vol_logdiff',
또는
'시가', '고가', '저가', '종가', 'Vol_logdiff', 'RSI14'
설정이 가장 좋은 예측
2가지 예측을 돌려서 더 좋은거 선택 ? 시간이 오래 걸리나 ?
"""
# ---- 전처리: NaN/inf 제거 및 피처 선택 ----
feature_cols = [
    '시가', '고가', '저가', '종가', 'Vol_logdiff',
    'RSI14', # ATR, CCI 같이 쓰지마

    # 'UltimateOsc', # 단독 위험, RSI 또는 이것만, CCI와 사용 금지
    # 'CCI14', # 단독 위험
    # 'ATR14',
    # 'STD20', 'UpperBand', 'LowerBand',
    # 'MA5_slope',
    # 'ROC12_pct',
]

# 4. 피쳐, 무한대 필터링
cols = [c for c in feature_cols if c in data.columns]  # 순서 보존
df = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
X_df = df.dropna() # X_df는 (정렬/결측처리된) 피처 데이터프레임, '종가' 컬럼 존재

idx_close = feature_cols.index('종가')

# 5. 스케일링, 시점 마스크
split = int(len(X_df) * 0.8)
scaler_X = StandardScaler().fit(X_df.iloc[:split])  # 원시 train 구간만, 중복 윈도우 때문에 같은 시점 행이 여러 번 들어가는 왜곡 방지
X_all = scaler_X.transform(X_df)                 # 전체 변환 (누수 없음)


# ↓ 여기서 X_all, Y_all을 '스케일된 X'로부터 만듦
X_tmp, Y, t0 = create_multistep_dataset(X_all, LOOK_BACK, N_FUTURE, idx=idx_close, return_t0=True)
Y_log = Y
print("    001 X_____.shape:", X_tmp.shape, "    Y_____.shape:", Y_log.shape)

# 시점 마스크로 분리
train_mask = (t0 + N_FUTURE - 1) < split
val_mask   = (t0 >= split)

X_train, Y_train = X_tmp[train_mask], Y_log[train_mask]
X_val,   Y_val   = X_tmp[val_mask],   Y_log[val_mask]

# ---- Train/Test split (스케일러는 Train으로만 fit) ----
Y_train = ensure_2d(Y_train)
Y_val  = ensure_2d(Y_val)
print("    002 X_tr__.shape:", X_train.shape, "    Y_tr__.shape:", Y_train.shape)

# ---- y 스케일링: Train으로만 fit ---- (타깃이 수익률이면 생략 가능)
scaler_y = StandardScaler().fit(Y_train)
y_train_scaled = scaler_y.transform(Y_train)
y_test_scaled  = scaler_y.transform(Y_val)
print("    003 y_tr_s.shape:", y_train_scaled.shape, "        y_te_s.shape:", y_test_scaled.shape)

# ---- 모델 ----
model = Sequential([
    LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(16, activation='relu'),
    Dense(N_FUTURE)
])

model.compile(optimizer=Adam(5e-4), loss=Huber(delta=1.0))
# model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
model.fit(
    X_train, y_train_scaled,
    batch_size=16, epochs=200, verbose=0, shuffle=False,
    validation_data=(X_val, y_test_scaled),
    callbacks=[early_stop, rlrop]
)

# ---- 예측 (스케일된 출력) ----
train_pred_scaled = model.predict(X_train, verbose=0) # (N, H)
test_pred_scaled  = model.predict(X_val, verbose=0)
print("    004 tr_p_s.shape:", train_pred_scaled.shape, "        te_p_s.shape:", test_pred_scaled.shape)

# inverse_transform는 2D를 기대
# y용 표준화 역변환: X-스케일로 되돌림
train_pred_xscale = scaler_y.inverse_transform(train_pred_scaled)   # (N_tr, 3)
test_pred_xscale  = scaler_y.inverse_transform(test_pred_scaled)    # (N_te, 3)
print("    005 __tr_p.shape:", train_pred_xscale.shape, "        __te_p.shape:", test_pred_xscale.shape)
"""
train_pred, test_pred **역스케일된 ‘가격 단위’**이고, shape은 (N, 3):
열 0: t+1 예측 종가
열 1: t+2 예측 종가
열 2: t+3 예측 종가
"""

# ---- 시각화 ----
# 정답도 X-스케일 (Y_train, Y_val) → 그대로 사용
y_train_xscale = Y_train   # (N_tr, 3)
y_val_xscale  = Y_val    # (N_te, 3)

# 실제 가격(원 단위)으로 변환 (종가 컬럼만 역변환)
n_features = X_train.shape[2]
y_train_price = inverse_close_matrix(y_train_xscale, scaler_X, n_features, idx_close)
y_val_price   = inverse_close_matrix(y_val_xscale,   scaler_X, n_features, idx_close)
train_pred_price = inverse_close_matrix(train_pred_xscale, scaler_X, n_features, idx_close)
test_pred_price  = inverse_close_matrix(test_pred_xscale,  scaler_X, n_features, idx_close)


# 3) h=1(다음날)만 비교해서 그리기
h = 0   # 0->t+1, 1->t+2, 2->t+3
plt.figure(figsize=(10,5))
offset = len(y_train_xscale)

# plt.plot(range(0, offset),               y_train_xscale[:, h], label=f'Train Actual (h={h+1})', linewidth=1)
# plt.plot(range(offset, offset+len(y_val_xscale)), y_val_xscale[:,  h], label=f'Test  Actual (h={h+1})',  linewidth=1, alpha=0.7)

# plt.plot(range(0, offset),               train_pred_scaled[:, h],   label=f'Train Pred   (h={h+1})')
# plt.plot(range(offset, offset+len(y_val_xscale)), test_pred_scaled[:,  h],   label=f'Test  Pred    (h={h+1})')

# 2번 역변환 > y축: 실제 종가
plt.plot(range(0, offset),                       y_train_price[:, h], label=f'Train Actual (h={h+1})', linewidth=1)
plt.plot(range(offset, offset+len(y_val_price)), y_val_price[:,  h],  label=f'Test  Actual (h={h+1})',  linewidth=1, alpha=0.7)

plt.plot(range(0, offset),                       train_pred_price[:, h], label=f'Train Pred (h={h+1})')
plt.plot(range(offset, offset+len(y_val_price)), test_pred_price[:,  h],  label=f'Test  Pred (h={h+1})')


plt.title(f'LSTM Prediction — look_back={LOOK_BACK}, horizon h={h+1}')
# plt.xlabel('Sample index'); plt.ylabel('Price'); plt.legend(); plt.tight_layout()
plt.xlabel('Sample index'); plt.ylabel('Price (KRW)'); plt.legend(); plt.tight_layout()
# plt.savefig('prediction_h2.png', dpi=150)
plt.show()






"""
Keras LSTM은 입력 shape을 (batch, timesteps, features) 로 받는다
X가 바로 LSTM의 입력에 맞는 3D 텐서

"""