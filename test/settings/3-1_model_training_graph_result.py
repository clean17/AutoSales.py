# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, drop_trading_halt_rows




# 혹시라도 1D가 되었다면 강제 2D 복원
def ensure_2d(y):
    return y.reshape(-1, 1) if y.ndim == 1 else y

# 스케일된 ‘종가’만 원 단위로 역변환
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



# 1. 데이터 수집
ticker = '000660' # 하이닉스
tickers = ['000660', '008970', '006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']

LOOK_BACK = 15
N_FUTURE = 3

for count, ticker in enumerate(tickers):
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
        'RSI14',
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

    # 시점 마스크로 분리
    train_mask = (t0 + N_FUTURE - 1) < split
    val_mask   = (t0 >= split)

    X_train, Y_train = X_tmp[train_mask], Y_log[train_mask]
    X_val,   Y_val   = X_tmp[val_mask],   Y_log[val_mask]

    # ---- Train/Test split (스케일러는 Train으로만 fit) ----
    Y_train = ensure_2d(Y_train)
    Y_val  = ensure_2d(Y_val)

    # ---- y 스케일링: Train으로만 fit ---- (타깃이 수익률이면 생략 가능)
    scaler_y = StandardScaler().fit(Y_train)
    y_train_scaled = scaler_y.transform(Y_train)
    y_test_scaled  = scaler_y.transform(Y_val)


    # ===== 공통 설정 =====
    input_shape = (X_train.shape[1], X_train.shape[2])
    early_stop  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlrop       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # ===== 권장 실험 구성 =====
    # - LSTM32: 가벼운 모델 (작은 데이터/빠른 실험)
    # - LSTM64: 밸런스형 (권장 기본)
    # - LSTM128: 깊고 넓게 (데이터/패턴이 충분할 때만)
    variants = {
        "LSTM": {
            "lstm_units": [32],
            "dropout": None,
            "dense_units": [16],
            "lr": 5e-4, "delta": 1.0
        },
        "LSTM32": {
            "lstm_units": [32,16],
            "dropout": 0.2,
            "dense_units": [16, 8],
            "lr": 3e-4, "delta": 1.0
        },
        "LSTM64": {
            "lstm_units": [64, 32],
            "dropout": 0.2,
            "dense_units": [32, 16],
            "lr": 3e-4, "delta": 1.0
        },
        "LSTM128": {
            "lstm_units": [128, 64],
            "dropout": 0.2,
            "dense_units": [64, 32],
            "lr": 3e-4, "delta": 1.0
        },
    }

    results = {}
    n_features = X_train.shape[2]

    for name, cfg in variants.items():
        print(f"[{name}] training...")
        model_v = create_lstm_model(
            input_shape, N_FUTURE,
            lstm_units=cfg["lstm_units"],
            dropout=cfg["dropout"],
            dense_units=cfg["dense_units"],
            lr=cfg["lr"], delta=cfg["delta"],
        )
        history = model_v.fit(
            X_train, y_train_scaled,
            batch_size=16, epochs=200, verbose=0, shuffle=False,
            validation_data=(X_val, y_test_scaled),
            callbacks=[early_stop, rlrop]
        )

        # 예측 (y-스케일)
        tr_pred_s = model_v.predict(X_train, verbose=0)
        va_pred_s = model_v.predict(X_val,   verbose=0)

        # y-스케일 -> X-스케일
        tr_pred_x = scaler_y.inverse_transform(tr_pred_s)
        va_pred_x = scaler_y.inverse_transform(va_pred_s)

        # X-스케일 -> 원 단위(종가만 역변환)
        tr_pred_p = inverse_close_matrix(tr_pred_x, scaler_X, n_features, idx_close)  # (N_tr, H)
        va_pred_p = inverse_close_matrix(va_pred_x, scaler_X, n_features, idx_close)  # (N_va, H)

        best_val = min(history.history.get('val_loss', [float('inf')]))
        results[name] = {
            "model": model_v,
            "train_pred_price": tr_pred_p,
            "val_pred_price":   va_pred_p,
            "best_val": best_val,
            "epochs_trained": len(history.history['loss']),
        }
        # print(f" -> best val_loss: {best_val:.6f} / epochs: {results[name]['epochs_trained']}")


    # ===== 최고 모델 선택 =====
    best_name = min(results.keys(), key=lambda k: results[k]["best_val"])
    best_model = results[best_name]["model"]
    print(f"\n[Best] {best_name} (val_loss={results[best_name]['best_val']:.6f})")

    # 실제 정답(원 단위)도 준비 (이미 위에서 계산했으면 재사용)
    y_train_price = inverse_close_matrix(Y_train, scaler_X, n_features, idx_close)
    y_val_price   = inverse_close_matrix(Y_val,   scaler_X, n_features, idx_close)


    h = 0  # 0->t+1, 1->t+2, 2->t+3
    offset = len(y_train_price)

    plt.figure(figsize=(11,5))
    plt.plot(range(0, offset),                       y_train_price[:, h], label=f'Actual Train (h={h+1})', linewidth=1)
    plt.plot(range(offset, offset+len(y_val_price)), y_val_price[:,  h], label=f'Actual Val   (h={h+1})', linewidth=1, alpha=0.8)

    for name in results:
        tp = results[name]["train_pred_price"]
        vp = results[name]["val_pred_price"]
        plt.plot(range(0, offset),                       tp[:,h], label=f'{name} Pred Train')
        plt.plot(range(offset, offset+len(y_val_price)), vp[:,h], label=f'{name} Pred Val')

    plt.title(f'Comparison — LSTM units 32/64/128, look_back={LOOK_BACK}, h={h+1}')
    plt.xlabel('Sample index'); plt.ylabel('Price (KRW)'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{ticker}_{best_name}.png', dpi=150)
    # plt.show()





"""
Keras LSTM은 입력 shape을 (batch, timesteps, features) 로 받는다
X가 바로 LSTM의 입력에 맞는 3D 텐서

"""