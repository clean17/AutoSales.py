'''
저점을 찾는 스크립트 (저점매수 + 반등 + 모멘텀)
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승

2025-02-02 되면 멈추는 조건 필요
'''
import matplotlib
matplotlib.use("Agg")  # ✅ 비인터랙티브 백엔드 (창 안 띄움)
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import lowscan_rules_83_25_260504 as rule0
import lowscan_rules as rule00
# import lowscan_rules_77_25_5_42 as rule1
# import lowscan_rules_80_25_4_42 as rule2
modules = [rule00]

# log_file = open("csv/output.log", "w", encoding="utf-8")
# sys.stdout = log_file
# sys.stderr = log_file
# print("이건 파일로 감")
# raise Exception("에러도 파일로 감")


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, low_weekly_check, extract_numbers_from_filenames, \
    sort_csv_by_today_desc, safe_read_pickle, safe_rate, to_float, round_float_features, pad_text

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
output_dir = 'F:\\5below20_test'
# output_dir = 'F:\\5below20'

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7
render_graph = 0

TEST_OFFSET = 60
START_OFFSET = 7      # 1이면 어제 기준부터 검증 가능.. 7일 검증을 사용하려면 7사용
END_OFFSET = 300      # 과거 300거래일까지 생성

if render_graph == 1:
    START_OFFSET = START_OFFSET + 8
# else:
#     END_OFFSET = END_OFFSET - TEST_OFFSET

def process_ticker(ticker, tickers_dict, i):
    results = []

    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[process_ticker] {stock_name} ({ticker}) 파일 없음")
        return results

    df = safe_read_pickle(filepath)

    # if render_graph == 0:
    #     df = df[:-TEST_OFFSET]  # 검증 데이터 분리 테스트

    # 데이터가 부족하면 패스
    if df is None or df.empty or len(df) < 70:
        return None

    # 2차 생성 feature
    df = add_technical_features(df)

    # 결측 제거
    df, _ = drop_sparse_columns(df, threshold=0.10, check_inf=True, inplace=True)

    # 거래정지/이상치 행 제거
    df, _ = drop_trading_halt_rows(df)

    # drop 이후 3차 생성
    df = add_technical_features(df)

    # 데이터가 부족하면 패스
    if df.empty or len(df) < 70:
        return None

    # 거래대금
    df["trading_value"] = df["종가"] * df["거래량"]

    # print(f"[idx={i}] {stock_name:<15} ({ticker})")  #  문자열을 왼쪽 정렬하고, 전체 너비를 15칸
    print(f"{pad_text(i, 4)} | {ticker} | {pad_text(stock_name, 26)}")  #  문자열을 왼쪽 정렬하고, 전체 너비를 15칸

    for idx in range(START_OFFSET, END_OFFSET + 1):
        res = process_one_with_df(df, idx, ticker, tickers_dict)
        if res is not None:
            results.append(res)

    return results

def process_one_with_df(df, idx, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    # 과거 데이터(data)와 / 검증 데이터(remaining_data)로 분리
    # [0:150](0~149), idx = 10 >> [0:140](0~139) / [140:](140~149)
    if idx != 0:
        data = df[:-idx]
        remaining_data = df[len(df)-idx:]
    else:
        data = df
        remaining_data = None

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 70:
        return

    # print('─────────────────────────────────────────────────────────────')
    # print(data.index[-1].date())
    # print('─────────────────────────────────────────────────────────────')

    ########################################################################

    # closes = data['종가'].values
    # 거래대금
    trading_value = data['trading_value']
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    today_tr_val_eok = today_tr_val / 100_000_000                    # 마지막 거래일 거래대금(억)
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 마지막 20일 거래대금 평균


    # ★★★★★ 20거래일 평균 거래대금 4억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if mean_prev20 / 100_000_000 < 4:
        return

    # 5일, 20일 이동평균선 없으면 패스
    REQUIRED_COLS = ["MA5", "MA20", "등락률"]

    for col in REQUIRED_COLS:
        if col not in data.columns:
            return

    """
    depth4 (진입 gate) >> 실패 줄이기
    - 오늘 +3% 이상
    - 최근 7일 하락 존재
    - 거래량 증가
    """
    # 오늘 제외 최근 7일 5일선이 20일선보다 계속 낮은데 -2.5% 하락이 있으면서 오늘 3.3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    signal = signal_any_drop(data, 7, 3.3, -2)
    # signal = signal_any_drop(data, 7, 2, -2.5)
    # signal = signal_any_drop(data, 8, 3.5, -3.0)
    if not signal:
        return

    ########################################################################
    # feature 만들기
    last = data.iloc[-1]
    
    # 오늘의 5일션 변동율
    _ma5_today          = data['MA5'].iloc[-1]
    _ma5_yesterday      = data['MA5'].iloc[-2]
    ma5_chg_rate        = safe_rate(_ma5_today, _ma5_yesterday)

    # ma5_recovery_rate_3d  = safe_rate(data["MA5"].iloc[-1], data["MA5"].iloc[-4])
    # ma20_recovery_rate_3d = safe_rate(data["MA20"].iloc[-1], data["MA20"].iloc[-4])
    # trend_signal_pct      = ma5_recovery_rate_3d - ma20_recovery_rate_3d * 0.5

    # 거래대금 변동률
    tr_value_ratio_3d = 0
    _tr_value_ratio_5d = 0
    tr_value_ratio_20d = 0
    
    if mean_prev3 > 0 and np.isfinite(mean_prev3):
        tr_value_ratio_3d = today_tr_val / (mean_prev3 + 1e-9)

    if mean_prev5 > 0 and np.isfinite(mean_prev5):
        _tr_value_ratio_5d = today_tr_val / (mean_prev5 + 1e-9)

    if mean_prev20 > 0 and np.isfinite(mean_prev20):
        tr_value_ratio_20d = today_tr_val / (mean_prev20 + 1e-9)

    # 최근 대비 오늘 거래대금 변동률 (중복)
    _tr_value_ratio       = tr_value_ratio_3d * 0.4 + _tr_value_ratio_5d * 0.6
    tr_val_rank_20d      = (trading_value.iloc[-20:] <= today_tr_val).mean()
    # tr_val_rank_60d    = (trading_value.iloc[-60:] <= today_tr_val).mean()

    # 최근 7거래일 최대 하락 (오늘 제외, 어제 포함)
    _recent_7_ret        = data['등락률'].iloc[-8:-1]
    max_drop_7d          = _recent_7_ret.min()

    # 오늘 등락률
    today_pct            = round(last['등락률'], 2)

    # 20일 최저/최고 대비 몇 % 올라왔는지
    _low_20d              = data['저가'].iloc[-20:].min()
    _low_60d              = data['저가'].iloc[-60:].min()
    _high_20d             = data["고가"].iloc[-20:].max()
    _high_60d             = data["고가"].iloc[-60:].max()
    dist_from_low_20d     = safe_rate(last['종가'], _low_20d)
    dist_from_low_60d     = safe_rate(last['종가'], _low_60d)

    # 이평선 대비 종가의 변동률
    dist_to_ma5           = safe_rate(last['종가'], last['MA5'])
    dist_to_ma20          = safe_rate(last['종가'], last["MA20"])

    # 5일선 자체의 저점 대비 회복률 (중복)
    # _ma5_low_20d          = data["MA5"].iloc[-20:].min()
    # dist_from_ma5_low_20d = safe_rate(last['MA5'], _ma5_low_20d)  # 중복

    # 이평선 1일, 3일 전의 갭
    _ma5_ma20_gap_1d_ago  = safe_rate(data["MA5"].iloc[-2], data["MA20"].iloc[-2])
    _ma5_ma20_gap_3d_ago  = safe_rate(data["MA5"].iloc[-4], data["MA20"].iloc[-4])
    _ma5_ma20_gap         = safe_rate(last["MA5"], last["MA20"])
    ma5_ma20_gap_chg_1d   = _ma5_ma20_gap - _ma5_ma20_gap_1d_ago
    # ma5_ma20_gap_chg_3d   = _ma5_ma20_gap - _ma5_ma20_gap_3d_ago  # 중복

    # 20일 저가~고가 구간에서 현재 종가가 어디쯤 있는지를 보는 값 (dist_to_ma20 상관 0.882)
    close_pos_20d        = (last["종가"] - _low_20d) / (_high_20d - _low_20d + 1e-9)

    # -------------------------------
    # 지표
    # -------------------------------
    # 하루 변화량 : 빠름, 노이즈 많음, 3일 : 신호는 늦엇지만 안정된 필터용
    _MACD_hist_1d        = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-2]
    MACD_hist_3d         = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-4]     # (AUC 0.545)
    ATR_pct = last["ATR14"] / last["종가"] * 100

    # -------------------------------
    # 필터
    # -------------------------------
    _dist_to_high_20d     = safe_rate(last['종가'], _high_20d)
    _BB_perc              = last['BB_perc']  # 중복
    _UltimateOsc          = last['UltimateOsc']
    _CCI14                = last['CCI14']
    _ADX14                = last['ADX14']
    _gap_pct              = (data['시가'].iloc[-1] / data['종가'].iloc[-2] - 1) * 100
    _vol_ratio_15_60      = round(data['등락률'].tail(15).std() / (data['등락률'].tail(60).std() + 1e-9), 3)   # 단기 변동성과 장기 변동성을 비교하는 비율
    _RSI_rebound          = data['RSI14'].iloc[-1] - data['RSI14'].iloc[-3]
    _close_pos_day        = (last["종가"] - last["저가"]) / (last["고가"] - last["저가"] + 1e-9)
    _rebound_power        = today_pct * (0.5 + _close_pos_day)
    _MACD_acc             = _MACD_hist_1d - (MACD_hist_3d / 3)  # MACD_hist_3d 보다 약함 (AUC 0.521)
    # MACD_hist_3d_atr_norm = MACD_hist_3d / (last["ATR14"] + 1e-9)
    _MACD_hist_3d_close_norm = MACD_hist_3d / (last["종가"] + 1e-9) * 100

    # -------------------------------
    # 변별력이 없는 피쳐
    # -------------------------------
    # dist_to_high_60d      = safe_rate(last['종가'], _high_60d)  # 너무 장거리

    # lower_wick_ratio = (min(last["시가"], last["종가"]) - last["저가"]) / (last["고가"] - last["저가"] + 1e-9)

    # 윗꼬리 비율 (값이 크면 고가에서 많이 내려옴, 매도압력)
    # upper_wick_ratio = (last["고가"] - max(last["시가"], last["종가"])) / (last["고가"] - last["저가"] + 1e-9)

    # ROC12_pct = last['ROC12_pct']

    # 캔들에서 몸통의 비율 (추세 강도)
    # body_ratio = abs(last["종가"] - last["시가"]) / (last["고가"] - last["저가"] + 1e-9)

    # 거래대금 변동률 (어제, 오늘)
    # tr_value_chg_1d      = today_tr_val / (trading_value.iloc[-2] + 1e-9)

    # MACD_rebound_power = (
    #         np.tanh(_MACD_acc / 50) * 0.65 +
    #         np.tanh(MACD_hist_3d / 100) * 0.35
    # )

    # 20일 최저점 발생일 대비 몇 일 지났는지 (19: 처음, 0: 오늘)
    # window               = data.iloc[-20:]
    # low_idx_pos          = window['저가'].values.argmin()   # 최솟값의 인덱스를 반환
    # days_since_low       = len(window) - 1 - low_idx_pos

    # 최근 5일 동안 상승일 거래대금이 하락일 거래대금보다 얼마나 강했는가, ma5_ma20_gap_chg_1d 유사, 거래대금이 작으면 설명력 부족
    # _recent_5d            = data.iloc[-5:]
    # _ret5                 = _recent_5d["등락률"]
    # _recent_tr_value      = _recent_5d["종가"] * _recent_5d["거래량"]
    #
    # _up_tr_value_5d       = _recent_tr_value[_ret5 > 0].sum()
    # _down_tr_value_5d     = _recent_tr_value[_ret5 < 0].sum()
    # up_down_tr_value_ratio_5d_log = np.log1p(
    #     _up_tr_value_5d / (_down_tr_value_5d + 1e-9)
    # )

    # 이미 많이 오른 종목 제거
    # _recent_runup        = data['등락률'].iloc[-5:-1].sum()

    # 시가 대비 종가가 얼마나 회복되었는가 (AUC 0.514 - 거의 랜덤)
    # intraday_return = (last["종가"] / last["시가"] - 1) * 100

    # 데드캣 패널티
    # _deadcat_penalty     = max(0, (-_drawdown_60d - 40) / 20) + max(0, -_dist_to_ma20 / 5)



    rule_features = {
        "today_pct": today_pct,
        "max_drop_7d": max_drop_7d,

        "dist_from_low_20d": dist_from_low_20d,
        "dist_from_low_60d": dist_from_low_60d,
        "dist_to_ma5": dist_to_ma5,
        # "dist_to_ma20": dist_to_ma20,  # close_pos_20d가 조합에 더 좋음 (AUC 0.524)
        "close_pos_20d": close_pos_20d,
        "ma5_ma20_gap_chg_1d": ma5_ma20_gap_chg_1d,
        # "ma5_recovery_rate_3d": ma5_recovery_rate_3d,  # ma5_ma20_gap_chg_1d, dist_to_ma20 겹침 (AUC 0.525)
        # "ma5_chg_rate": ma5_chg_rate,  # ma5_ma20_gap_chg_1d 상관 0.930

        "today_tr_val_eok": today_tr_val_eok,
        "tr_val_rank_20d": tr_val_rank_20d,

        "MACD_hist_3d": MACD_hist_3d,
        "ATR_pct": ATR_pct,
    }


    ############################  deadcat_filter  ###########################

    # 성공 최저 확인용
    other_rule_features = {
        "_tr_value_ratio": _tr_value_ratio,
        "_tr_value_ratio_5d": _tr_value_ratio_5d,
        "_dist_to_high_20d": _dist_to_high_20d,
        "_BB_perc": _BB_perc,
        "_UltimateOsc": _UltimateOsc,
        "_CCI14": _CCI14,
        "_ADX14": _ADX14,
        "_gap_pct": _gap_pct,
        "_vol_ratio_15_60": _vol_ratio_15_60,
        "_RSI_rebound": _RSI_rebound,
        "_rebound_power": _rebound_power,
        "_MACD_hist_1d": _MACD_hist_1d,
        "_MACD_acc": _MACD_acc,
        "_MACD_hist_3d_close_norm": _MACD_hist_3d_close_norm,
    }

    rule_features.update(other_rule_features)

    # if ATR_pct < 2.463:
    #     return
    # if today_tr_val_eok < 0.594:
    #     return
    # if close_pos_20d < 0.043:
    #     return
    # if dist_to_ma5 < -16.293:
    #     return
    # if dist_from_low_20d < 3.302:
    #     return
    # if ma5_ma20_gap_chg_1d < -5.933:
    #     return
    # if today_tr_val_eok < 0.594:
    #     return
    # if _MACD_hist_3d_close_norm < -5.386:
    #     return
    # if _BB_perc < -0.117:
    #     return
    # if _dist_to_high_20d < -72.546:
    #     return
    # if _UltimateOsc < 12.458:
    #     return
    # if _tr_value_ratio_5d < 0.073:
    #     return
    # if _tr_value_ratio < 0.066:
    #     return
    # if _CCI14 < -261.263:
    #     return
    # if _ADX14 < 7.555:
    #     return
    # if _gap_pct < -9.187:
    #     return
    # if _vol_ratio_15_60 < 0.246:
    #     return
    # if _RSI_rebound < -15.247:
    #     return
    # if _rebound_power < 1.823:
    #     return
    # if _MACD_hist_1d < -553.4:
    #     return
    # if _MACD_acc < -479.478:
    #     return

    ########################################################################

    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)
    m_current = m_data['종가'].iloc[-1]                               # 오늘 종가, 검증 데이터로 잘랐다면 검증 직전까지의 마지막 값 (수익률 분석 용도)

    predict_str = ''
    validation_chg_rate = 0
    validation_chg_rate1 = 0
    validation_chg_rate2 = 0
    validation_chg_rate3 = 0
    validation_chg_rate4 = 0
    validation_chg_rate5 = 0
    validation_chg_rate6 = 0
    validation_chg_rate7 = 0
    validation_chg_rate_min = 0

    # 검증 데이터 (마지막 n일)
    if remaining_data is not None:
        r_closes = remaining_data['종가'].iloc[:7].reset_index(drop=True)  # Series 인덱스 새로
        r_closes = r_closes.reindex(range(7))      # 0~6 없으면 NaN으로 채움
        r_max = r_closes.max(skipna=True)          # 결측치(NaN)를 무시하고 계산
        r_min = r_closes.max(skipna=True)          # 결측치(NaN)를 무시하고 계산

        r1, r2, r3, r4, r5, r6, r7 = (r_closes.iloc[i] for i in range(7))

        # 마지막 종가로부터 n일차 동안의 수익률
        validation_chg_rate  = round(safe_rate(r_max, m_current), 2)
        validation_chg_rate1 = round(safe_rate(r1, m_current), 2)
        validation_chg_rate2 = round(safe_rate(r2, m_current), 2)
        validation_chg_rate3 = round(safe_rate(r3, m_current), 2)
        validation_chg_rate4 = round(safe_rate(r4, m_current), 2)
        validation_chg_rate5 = round(safe_rate(r5, m_current), 2)
        validation_chg_rate6 = round(safe_rate(r6, m_current), 2)
        validation_chg_rate7 = round(safe_rate(r7, m_current), 2)
        validation_chg_rate_min = round(safe_rate(r_min, m_current), 2)

        predict_str = '상승'
        if validation_chg_rate < VALIDATION_TARGET_RETURN:
            predict_str = '미달'

    # result = low_weekly_check(m_data)


    ########################################################################

    row = {
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "predict_str": predict_str,                      # 상승/미달
        "idx": idx,

        "validation_chg_rate": validation_chg_rate,      # 검증 등락률
        "validation_chg_rate1": validation_chg_rate1,    # 검증 등락률
        "validation_chg_rate2": validation_chg_rate2,    # 검증 등락률
        "validation_chg_rate3": validation_chg_rate3,    # 검증 등락률
        "validation_chg_rate4": validation_chg_rate4,    # 검증 등락률
        "validation_chg_rate5": validation_chg_rate5,    # 검증 등락률
        "validation_chg_rate6": validation_chg_rate6,    # 검증 등락률
        "validation_chg_rate7": validation_chg_rate7,    # 검증 등락률
    }
    row.update(rule_features)
    row = round_float_features(row)
    row["ticker"] = ticker  # 종목코드 숫자 float 처리 되어서 밖으로 뺌

    # 처음으로 수익률 뚫는 날, 조건이 뚫을 수 있는지 확인
    vals = [
        validation_chg_rate1,
        validation_chg_rate2,
        validation_chg_rate3,
        validation_chg_rate4,
        validation_chg_rate5,
        validation_chg_rate6,
        validation_chg_rate7,
    ]

    hit_day = None

    for i, v in enumerate(vals, start=1):
        if v >= VALIDATION_TARGET_RETURN:
            hit_day = i
            break

    row["hit_day"] = hit_day if hit_day is not None else 0
    row["is_success"] = 1 if hit_day is not None else 0
    row["target"] = VALIDATION_TARGET_RETURN


    return {
        "row": row,
    }



if __name__ == "__main__":
    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 7_find_low_point.py...')
    print('=== 7일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -2.5%이 존재 + 오늘 3.3% 이상 상승 ===')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())
    # tickers = ['103140']  # 풍산
    # tickers = extract_numbers_from_filenames(directory = r'D:\5below20_test\4퍼', isToday=False)


    rows = []            # 결과 종목 데이터 저장

    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            futures = [
                executor.submit(process_ticker, ticker, tickers_dict, i)
                for i, ticker in enumerate(tickers, start=1)
            ]

            for f in as_completed(futures):
                try:
                    results = f.result()
                    if results is None:
                        continue
                except Exception as e:
                    print("worker error:", e)
                    raise


                for res in results:
                    row = res["row"]
                    rows.append(row)


        rows_sorted = sorted(rows, key=lambda row: row['today'])


        if len(rows) > 0:
            df = pd.DataFrame(rows)

            # df["MACD_hist_3d_rank"] = (
            #     df.groupby("today")["MACD_hist_3d"]
            #     .rank(pct=True)
            #     .round(4)
            # )
            # MACD_hist_3d_rank 생성 후 제거
            # df.drop(columns=["MACD_hist_3d"], inplace=True)

            if render_graph == 1:
                mask = pd.Series(False, index=df.index)
                matched_rule_map = {idx: [] for idx in df.index}

                for mod in modules:
                    conditions = mod.build_conditions(df)

                    for name in mod.RULE_NAMES:
                        if name not in conditions:
                            continue

                        cond = conditions[name].fillna(False)

                        # 하나라도 만족하면 통과
                        mask |= cond

                        # 어떤 룰에 걸렸는지 기록
                        for idx in df.index[cond]:
                            matched_rule_map[idx].append(name)

                df["matched_rules"] = df.index.map(
                    lambda idx: ",".join(matched_rule_map[idx])
                )

                selected = df[mask].copy()

                total_cnt = len(selected)
                up_cnt = int(selected["is_success"].sum())
                shortfall_cnt = total_cnt - up_cnt
                total_up_rate = up_cnt / total_cnt * 100 if total_cnt else 0

                print(f"\n룰 통과 수: {total_cnt}")
                print(f"룰 통과 후 성공률: {total_up_rate:.2f}% ({up_cnt} / {total_cnt})")

            if render_graph == 0:
                # CSV 저장
                df.to_csv('csv/low_result_7.csv', index=False)  # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
                saved = sort_csv_by_today_desc(
                    in_path=r"csv/low_result_7.csv",
                    out_path=r"csv/low_result_7_desc.csv",
                )
                print("saved:", saved)


        if render_graph == 1:
            plot_jobs = []

            for _, row in selected.iterrows():
                ticker = row["ticker"]
                stock_name = row["stock_name"]

                filepath = os.path.join(pickle_dir, f"{ticker}.pkl")
                origin = safe_read_pickle(filepath)

                if origin is None or origin.empty:
                    continue

                origin = add_technical_features(origin)
                origin, _ = drop_sparse_columns(
                    origin,
                    threshold=0.10,
                    check_inf=True,
                    inplace=True
                )
                origin, _ = drop_trading_halt_rows(origin)
                origin = add_technical_features(origin)

                signal_day = pd.to_datetime(row["today"])

                # 기본값: idx == 0이면 전체 origin 사용
                chart_data = origin.copy()

                if idx != 0:
                    if signal_day not in origin.index:
                        continue

                    signal_pos = origin.index.get_loc(signal_day)

                    # today 이후 START_OFFSET 거래일까지 포함
                    chart_end_pos = signal_pos + START_OFFSET + 1
                    chart_end_pos = min(chart_end_pos, len(origin))

                    chart_data = origin.iloc[:chart_end_pos].copy()

                if chart_data.empty or len(chart_data) < 70:
                    continue

                today = row["today"].replace("-", "")

                title = (
                    f"{today} {stock_name} [{ticker}] "
                    f"{row['today_pct']}% Daily Chart - "
                    f"{row['predict_str']} {row['validation_chg_rate']}% "
                    f"rules={row['matched_rules'][:80]}"
                )

                final_file_name = (
                    f"{today} {stock_name} [{ticker}] "
                    f"{row['today_pct']}%_{row['predict_str']}.webp"
                )

                save_path = os.path.join(output_dir, final_file_name)

                plot_jobs.append({
                    "origin": chart_data,
                    "today": today,
                    "title": title,
                    "save_path": save_path,
                })


            for job in plot_jobs:
                fig = plt.figure(figsize=(14, 16), dpi=150)
                gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

                ax_d_price = fig.add_subplot(gs[0, 0])
                ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
                ax_w_price = fig.add_subplot(gs[2, 0])
                ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

                plot_candles_daily(
                    job["origin"],
                    show_months=4,
                    title=job["title"],
                    ax_price=ax_d_price,
                    ax_volume=ax_d_vol,
                    date_tick=5,
                    today=job["today"],
                )

                plot_candles_weekly(
                    job["origin"],
                    show_months=12,
                    title="Weekly Chart",
                    ax_price=ax_w_price,
                    ax_volume=ax_w_vol,
                    date_tick=5,
                )

                plt.tight_layout()
                plt.savefig(
                    job["save_path"],
                    format="webp",
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0.1
                )
                plt.close()


            if len(selected) > 0 and render_graph == 1:
                print("\n그래프 생성 완료")


        end = time.time()     # 끝 시간(초)
        elapsed = end - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
        # log_file.close()
        # print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

    except KeyboardInterrupt:
        print("\n사용자 중지 요청 감지. 작업을 종료합니다.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise