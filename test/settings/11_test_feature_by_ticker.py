import matplotlib
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 현재 파일에서 2단계 위 폴더 경로 구하기
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# print('BASE_DIR', BASE_DIR)
sys.path.append(BASE_DIR)

from utils import create_multistep_dataset, add_technical_features




def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en


"""
feature_cols = ['Close','Volume']
look_back = 3

# 결과:
['Close_t-2','Volume_t-2',
 'Close_t-1','Volume_t-1',
 'Close_t0', 'Volume_t0']
"""
def build_flat_feature_names(feature_cols, look_back):
    names = []
    for t in range(look_back):
        names += [f"{f}_t{-look_back + t + 1}" for f in feature_cols]
    return names

"""
        시계열 데이터를 LSTM 학습에 바로 쓸 수 있도록 준비하는 파이프라인
"""
def prepare_lstm_data(
        df: pd.DataFrame,
        feature_cols,
        look_back: int,
        horizon: int,
        target_col: str = "Close",
        scaler_cls = MinMaxScaler,         # 또는 StandardScaler
        scaler_kwargs = {"feature_range": (0,1)},
        val_ratio: float = 0.1,
        dropna: bool = True
):
    """
    1) 정렬 → 2) 결측 제거 → 3) 타깃 생성 → 4) 시간순 split → 5) train에만 스케일 fit → 6) windowing
    타깃은 예시로 1-step ahead '가격 수익률' 사용. (원하면 가격 자체/다중스텝 등으로 수정)
    """
    df = df.copy().sort_index()
    if dropna:
        df = df.dropna(subset=feature_cols + [target_col])

    # === 타깃 정의(예시): 1-step ahead 수익률 ===
    #   horizon을 길게 쓰고 싶으면 shift(-horizon) 후 윈도 loop에서 i+h 쓰면 됩니다.
    target_ret = (df[target_col].shift(-1) / df[target_col] - 1.0)

    # 마지막 NaN(target) 제거
    valid_idx = target_ret.notna()
    df = df.loc[valid_idx]
    target_ret = target_ret.loc[valid_idx]

    # === 특성 행렬 ===
    X_full = df[feature_cols].values.astype(float)
    y_full = target_ret.values.astype(float)

    # === 시간순 분리 ===
    split = int(len(df) * (1 - val_ratio))
    X_train_raw, X_val_raw = X_full[:split], X_full[split:]
    y_train_raw, y_val_raw = y_full[:split], y_full[split:]

    # === 스케일링 (train으로만 fit) ===
    scaler = scaler_cls(**scaler_kwargs) if scaler_kwargs is not None else scaler_cls()
    X_train_2d = scaler.fit_transform(X_train_raw)
    X_val_2d   = scaler.transform(X_val_raw)

    # === 윈도잉 ===
    # 주의: train/val 각각의 구간 내에서만 window를 만들어 **경계가 섞이지 않게** 합니다.
    def windowize_block(Xblock, yblock, look_back, horizon):
        Xb, Yb = [], []
        T, F = Xblock.shape
        last_idx = T - horizon
        for i in range(look_back - 1, last_idx):
            Xb.append(Xblock[i - look_back + 1:i + 1, :])
            Yb.append(yblock[i + horizon])  # horizon=3
        return np.asarray(Xb), np.asarray(Yb)

    X_train, y_train = windowize_block(X_train_2d, y_train_raw, look_back, horizon)
    X_val,   y_val   = windowize_block(X_val_2d,   y_val_raw,   look_back, horizon)

    # === 플랫 feature names (RF/Permutation importance 용) ===
    flat_names = build_flat_feature_names(feature_cols, look_back)

    # 무결성 체크
    assert X_train.shape[2] == len(feature_cols)
    assert len(flat_names)  == look_back * len(feature_cols)

    meta = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "flat_feature_names": flat_names,
        "look_back": look_back,
        "horizon": horizon,
        "split_index": split,
        "target_col": target_col,
    }
    return X_train, y_train, X_val, y_val, meta

def select_top_base_features_via_permutation(
        data,                      # add_technical_features 적용된 DataFrame
        feature_cols,              # 원 후보 지표 리스트 (당신이 주신 feature_cols)
        look_back=60,
        horizon=1,
        target_col="Close",        # 또는 "종가"
        scaler_cls=None,           # 예: MinMaxScaler
        scaler_kwargs=None,
        val_ratio=0.1,
        top_k=10,
        n_estimators=500,
        n_repeats=10,
        random_state=42
):
    """
    1) prepare_lstm_data로 시계열 전처리
    2) RF 학습 후 검증셋 permutation importance
    3) flat 이름('feat_t-5') → 원 피처('feat')로 집계하여 합산 중요도 계산
    4) 합산 기준 Top-K 원 피처 반환
    """
    # --- 1) 데이터 준비 (train/val split, scaling, windowing)
    X_train, y_train, X_val, y_val, meta = prepare_lstm_data(
        data, feature_cols, look_back=look_back, horizon=horizon,
        target_col=target_col,
        scaler_cls=scaler_cls, scaler_kwargs=(scaler_kwargs or {}),
        val_ratio=val_ratio, dropna=True
    )

    # --- 2) RF 학습 (flat으로 변환)
    n_tr, T, F = X_train.shape
    n_va = X_val.shape[0]
    X_rf_tr = X_train.reshape(n_tr, T * F)
    X_rf_val = X_val.reshape(n_va, T * F)

    rf = RandomForestRegressor(n_estimators=n_evaluators if (n_evaluators:=n_estimators) else 500,
                               random_state=random_state, n_jobs=-1)
    rf.fit(X_rf_tr, y_train)

    # --- 3) Permutation Importance (검증셋)
    perm = permutation_importance(
        rf, X_rf_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    imps = perm.importances_mean                 # (T*F,)
    flat_names = meta['flat_feature_names']      # 길이 T*F, 예: 'ROC12_pct_t-8'

    # --- 4) flat -> base feature로 집계
    base_names = []
    for n in flat_names:
        m = re.search(r'(.*)_t-?\d+', n)
        base_names.append(m.group(1) if m else n)

    agg = {}
    for b, imp in zip(base_names, imps):
        # 음수 중요도는 0으로 클리핑(선택): 안정적 집계를 위해
        agg[b] = agg.get(b, 0.0) + max(0.0, float(imp))

    agg_s = pd.Series(agg).sort_values(ascending=False).round(4)

    # --- 5) Top-K 원 피처 선택
    top_features = list(agg_s.index[:top_k])

    # 부가 정보(원피처 중요도 테이블)도 함께 반환
    return {
        "top_features": top_features,      # 최종 선택 원 피처
        "agg_importance": agg_s,           # 원 피처 합산 중요도 Series
        "meta": meta,                      # prepare_lstm_data 메타
        "rf_model": rf,                    # 학습한 RF
        "perm": perm                       # permutation 결과 원본
    }










PREDICTION_PERIOD = 3
LOOK_BACK = 15
H = 3

# tickers = ['MNKD', 'ESPR', 'ALKS', 'LASR', 'TLRY', 'TSLA', 'SNDL', 'INSG', 'SABR', 'TBPH', 'VFF', 'AVDL', 'EVLV']
tickers = ['006490', '042670', '023160', '006800', '323410', '009540', '058970', '034020', '079550', '358570', '000155', '035720', '00680K', '035420', '012510']

# 2단계 위 디렉토리를 루트 디렉토리로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print('ROOT_DIR', ROOT_DIR)
# pickle_dir = os.path.join(ROOT_DIR, 'pickle_us')
pickle_dir = os.path.join(ROOT_DIR, 'pickle')


for count, ticker in enumerate(tickers):
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)

    data = add_technical_features(df)
    # print(len(data))

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    # print("NaN 개수")
    # print(data.isna().sum())
    # print("\ninf/-inf 개수")
    # print(np.isinf(data).sum())
    # print("NaN 비율(%)")
    # total = len(data)
    # print((data.isna().sum() / total * 100).round(2))
    # print("\ninf/-inf 비율(%)")
    # print((np.isinf(data).sum() / total * 100).round(2))
    '''
        일반적으로 10% 이상 NaN/inf면 문제
        만약 “NaN이 1~2개밖에 없음” → 그냥 0이나 평균값으로 대체해도 무방
        “절반이 NaN/inf다” → 그 feature는 제거하는 것이 안전
    '''
    # NaN/inf를 자동 제거하려면
    # NaN/inf가 전체의 10% 이상인 feature만 자동 drop
    # threshold = 0.1  # 10%
    # cols_to_drop = [
    #     col for col in data.columns
    #     if (data[col].isna().mean() > threshold) or (np.isinf(data[col]).mean() > threshold)
    # ]
    # print("Drop candidates:", cols_to_drop)
    # print('')


    # feature_cols = ['시가', '고가', '저가', '종가', '거래량', 'MA20', 'UpperBand', 'LowerBand', 'PBR']
    feature_cols = [
        col_o, col_l, col_h, col_c,
        'Vol_logdiff',
        'ma10_gap',
        'MA5_slope',
    ]

    X_train, y_train, X_val, y_val, meta = prepare_lstm_data(
        data, feature_cols, look_back=LOOK_BACK, horizon=H,
        target_col=col_c,                 # 또는 "ROC12_pct" 같은 수익률 기반 타깃을 직접 쓰셔도 됩니다.
        scaler_cls=MinMaxScaler, scaler_kwargs={"feature_range": (0,1)},
        val_ratio=0.1, dropna=True
    )

    result = select_top_base_features_via_permutation(
        data=data,
        feature_cols=feature_cols,
        look_back=LOOK_BACK,
        horizon=H,                     # 3일 뒤만 예측이면, prepare_lstm_data 내부 타깃 인덱스도 i+H 사용하도록 수정되어 있어야 해요
        target_col=col_c,              # 또는 "Close"
        scaler_cls=MinMaxScaler,
        scaler_kwargs={"feature_range": (0,1)},
        val_ratio=0.1,
        top_k=10,
        n_estimators=500,
        n_repeats=10,
        random_state=42
    )
    # print("top_features : ", result["top_features"])
    # print("agg_importance : ",result["agg_importance"])

    agg_s = result["agg_importance"]         # Series: feature -> total_importance (clipped sum)
    thresh = 0.01                            # 임계치 (너무 높으면 비어질 수 있어요)
    sel_agg = agg_s[agg_s >= thresh].round(4)
    print("=== 합산 중요도 (>= 0.01) ===")
    print(sel_agg.to_string())

    continue

    # 3) RF 중요도 분석 (검증 세트 기준)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    n_samples, look_back, n_features = X_val.shape
    X_rf_val = X_val.reshape(n_samples, look_back * n_features)
    X_rf_tr  = X_train.reshape(X_train.shape[0], look_back * n_features)

    """
            랜덤포레스트 회귀 모델을 학습시키는 단계
            500개의 의사결정트리 사용
            random_state=42 → 재현 가능성 확보
            n_jobs=-1 → CPU 코어 전부 사용해 학습 속도 ↑
            X_rf_tr: 학습 피처 (flatten된 look_back × feature)
            y_train: 타깃 벡터
    """
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_rf_tr, y_train)



    """
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
            
            검증 데이터에서 특정 피처 값을 랜덤으로 섞어서 모델 성능이 얼마나 떨어지는지 측정
            성능 하락이 크면 → 그 피처가 예측에 중요하다는 뜻
            n_repeats=10 → 무작위 섞기 실험을 10번 반복해 평균/분산 계산
            장점: 스케일에 덜 민감하고, 트리 기반 feature_importances_보다 신뢰성이 높음
    """
    perm = permutation_importance(rf, X_rf_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

    """
            perm.importances_mean: 각 피처별 평균 중요도
            np.argsort(...)[::-1]: 중요도가 큰 순서대로 인덱스를 정렬
            [:20]: 상위 20개 피처만 뽑음
    """
    # idx = np.argsort(perm.importances_mean)[::-1][:100]
    # for i in idx:
    #     print(f"{meta['flat_feature_names'][i]}: {perm.importances_mean[i]:.4f}")
    #     # ROC12_pct_t-8 (0.0707) : 12일 기준 변동률(ROC), 8일 전 값이 가장 큰 영향력.


    # ---- 플랫 기준 Top-K 선택 ----
    K = 10
    imps = perm.importances_mean
    names = meta['flat_feature_names']
    top_idx = np.argsort(imps)[::-1][:K]
    sel_names_flat = [names[i] for i in top_idx]

    # 원 피처명만 뽑고 lag 정보 분리
    base_names = [n.rsplit('_t', 1)[0] for n in sel_names_flat]
    lags       = [n.rsplit('_t', 1)[1] for n in sel_names_flat]
    print("선택된 (피처, lag):", list(zip(base_names, lags)))

