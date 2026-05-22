'''
저점을 찾는 스크립트 (저점매수 + 반등 + 모멘텀)
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승

2025-02-02 되면 멈추는 조건 필요
'''
import matplotlib
matplotlib.use("Agg")  # 비인터랙티브 백엔드 (창 안 띄움)
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import lowscan_rules_v1 as rule1

modules = [rule1]

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
    sort_csv_by_today_desc, safe_read_pickle, safe_rate, to_float, round_float_features, pad_text, \
    first_reach_day_from_rates, make_trade_labels

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
output_dir = 'F:\\5below20_test'
# output_dir = 'F:\\5below20'


CSV_PATH = r"csv/low_result_7.csv"
SORTED_CSV_PATH = r"csv/low_result_7_desc.csv"

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7
EXECUTION_RATIO = 1
# 최고가 판매는 힘드니까 보수적으로
REQUIRED_HIGH_RETURN = VALIDATION_TARGET_RETURN * EXECUTION_RATIO
VALIDATION_DAYS = 7
STOP_LOSS = -6
render_graph = 1

# TEST_OFFSET = 90      # 2026부터 테스트
TEST_OFFSET = 0
START_OFFSET = 7      # 1이면 어제 기준부터 검증 가능.. 7일 검증을 사용하려면 7사용
END_OFFSET = 40      # 2024년 ~ 2025년 학습

if render_graph == 1:
    START_OFFSET = START_OFFSET + 3
else:
    END_OFFSET = END_OFFSET - TEST_OFFSET  # df = df[:-TEST_OFFSET] 테스트 데이터를 자른만큼



def process_ticker(ticker, tickers_dict, i):
    results = []
    stock_name = tickers_dict.get(ticker).get("stock_name", 'Unknown Stock')
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[process_ticker] [idx={i}] {ticker} 파일 없음")
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

    last_date = df.index[-1].strftime("%Y-%m-%d")
    first_index = min(len(df), END_OFFSET)
    first_date = df.index[-first_index].strftime("%Y-%m-%d")
    # print(f"[idx={i}] {stock_name:<15} ({ticker})")  #  문자열을 왼쪽 정렬하고, 전체 너비를 15칸
    print(f"{pad_text(i, 4)} | {first_date} - {last_date} | {ticker} | {pad_text(stock_name, 26)} | {len(df)}")  #  문자열을 왼쪽 정렬하고, 전체 너비를 15칸

    for idx in range(START_OFFSET, END_OFFSET + 1):
        res = process_one_with_df(df, idx, ticker, tickers_dict)
        if res is not None:
            results.append(res)

    return results

def process_one_with_df(df, idx, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker).get("stock_name", 'Unknown Stock')

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

    ########################################################################

    # 거래대금
    trading_value = data['trading_value']

    today_tr_val = round(trading_value.iloc[-1], 2)                  # 기준일 거래대금
    today_tr_val_eok = today_tr_val / 100_000_000                    # 기준일 거래대금(억)
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 기준일 전 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 기준일 전 5일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 기준일 전 20일 거래대금 평균

    _mean_prev5_eok = mean_prev5 / 100_000_000
    _mean_prev20_eok = mean_prev20 / 100_000_000

    # ★★★★★ 20거래일 평균 거래대금 4억보다 작으면 패스
    if mean_prev20 / 100_000_000 < 4:
        return None

    # 5일, 20일 이동평균선 없으면 패스
    REQUIRED_COLS = ["MA5", "MA20", "등락률"]

    for col in REQUIRED_COLS:
        if col not in data.columns:
            return None

    # 기준일 5일선
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산
    ma5_chg_rate = (ma5_today - ma5_yesterday) / ma5_yesterday * 100

    """
    depth4 (진입 gate) >> 실패 줄이기
    - 기준일 +3% 이상
    - 최근 7일 하락 존재
    - 거래량 증가
    """
    # 기준일 제외 최근 7일 5일선이 20일선보다 계속 낮은데 -2.5% 하락이 있으면서 기준일 3.3% 상승
    signal = signal_any_drop(data, 7, 3.3, -2.5)

    if not signal:
        return None

    ########################################################################
    # feature 만들기
    ########################################################################

    # 최근 변동성
    last15_ret = data['등락률'].tail(15)
    last20_ret = data['등락률'].tail(20)
    last30_ret = data['등락률'].tail(30)

    vol15 = last15_ret.std()
    vol30 = last30_ret.std()

    # 양봉 비율
    pos20_ratio = (last20_ret > 0).mean()

    # 추가 독립 피쳐
    def local_to_float(x):
        return float(x) if pd.notna(x) else np.nan

    last = data.iloc[-1]
    close_pos = round(local_to_float(last.get("close_pos")), 4)

    ########################################################################

    m_data = data[-60:]  # 뒤에서 60개

    m_closes = m_data['종가']
    m_max = m_closes.max()
    m_min = m_closes.min()
    m_current = m_closes.iloc[-1]

    three_m_chg_rate = (m_max - m_min) / m_min * 100        # 최근 3개월 동안의 등락률
    today_chg_rate = (m_current - m_max) / m_max * 100      # 최근 3개월 최고 대비 기준일 등락률

    result = low_weekly_check(m_data)

    if result["ok"]:
        # ★★★★★ 저번주 대비 이번주 증감률 -1%보다 낮으면 패스
        if result["is_drop_more_than_minus1pct"]:
            # return None
            pass

    ########################################################################

    ma5_chg_rate = round(ma5_chg_rate, 4)
    vol15 = round(vol15, 4)
    vol30 = round(vol30, 4)
    pos20_ratio = round(pos20_ratio * 100, 4)
    mean_prev3 = round(mean_prev3, 4)
    today_tr_val = round(today_tr_val, 4)
    three_m_chg_rate = round(three_m_chg_rate, 4)
    today_chg_rate = round(today_chg_rate, 4)
    pct_vs_lastweek = round(result['pct_vs_lastweek'], 4)
    pct_vs_last4week = round(result['pct_vs_last4week'], 4)
    today_pct = round(data.iloc[-1]['등락률'], 2)

    # --- build_conditions()가 참조하는 컬럼들을 data에 주입 (스칼라 → 컬럼 브로드캐스트) ---
    rule_features = {
        "ma5_chg_rate": ma5_chg_rate,
        "vol15": vol15,
        "vol30": vol30,
        "pos20_ratio": pos20_ratio,
        "today_tr_val": today_tr_val,
        "mean_prev3": mean_prev3,
        "three_m_chg_rate": three_m_chg_rate,
        "today_chg_rate": today_chg_rate,
        "pct_vs_lastweek": pct_vs_lastweek,
        "pct_vs_last4week": pct_vs_last4week,
        "today_pct": today_pct,
        "close_pos": close_pos,
    }

    ########################################################################

    row = {
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "idx": idx,
    }

    # 검증 데이터 (마지막 n일)
    if remaining_data is not None:
        r_closes = remaining_data['종가'].iloc[:VALIDATION_DAYS].reset_index(drop=True)  # Series 인덱스 새로
        r_highs  = remaining_data["고가"].iloc[:VALIDATION_DAYS].reset_index(drop=True)
        r_lows   = remaining_data["저가"].iloc[:VALIDATION_DAYS].reset_index(drop=True)
        r_opens  = remaining_data["시가"].iloc[:VALIDATION_DAYS].reset_index(drop=True)

        r_closes = r_closes.reindex(range(VALIDATION_DAYS))      # 0~6 없으면 NaN으로 채움
        r_highs  = r_highs.reindex(range(VALIDATION_DAYS))
        r_lows   = r_lows.reindex(range(VALIDATION_DAYS))
        r_opens  = r_opens.reindex(range(VALIDATION_DAYS))

        # 익절 가능 최대 수익률: 고가 기준
        validation_high_rate_max     = safe_rate(r_highs.max(skipna=True), m_current)  # 결측치(NaN)를 무시하고 계산
        # validation_close_rate_max    = safe_rate(r_closes.max(skipna=True), m_current) # 익절 기준: 종가

        # 손절 위험 최소 수익률: 저가 기준
        validation_low_rate_min      = safe_rate(r_lows.min(skipna=True), m_current)

        # 일별 종가 수익률: 참고용
        close_rates = [
            safe_rate(r_closes.iloc[i], m_current)
            for i in range(VALIDATION_DAYS)
        ]

        # 일별 고가 수익률: 익절 판정용
        high_rates = [
            safe_rate(r_highs.iloc[i], m_current)
            for i in range(VALIDATION_DAYS)
        ]

        # 일별 저가 수익률: 손절 판정용
        low_rates = [
            safe_rate(r_lows.iloc[i], m_current)
            for i in range(VALIDATION_DAYS)
        ]

        # 시가
        open_rates = [
            safe_rate(r_opens.iloc[i], m_current)
            for i in range(VALIDATION_DAYS)
        ]


        max_high_7d = validation_high_rate_max

        if max_high_7d < 4:     # 4% 미만
            target_class = 0
        elif max_high_7d < 7:   # 4~7
            target_class = 1
        elif max_high_7d < 12:  # 7~12
            target_class = 2
        else:                   # 12+
            target_class = 3


        trade_labels = make_trade_labels(
            high_rates=high_rates,
            low_rates=low_rates,
            close_rates=close_rates,
            stop_loss=STOP_LOSS
        )


        """
        class_0~3 구간 분류
          1. class_0 제거 룰을 먼저 만든다. ++ 성공보다 이탈을 먼저하는 종목 룰
          2. class_1은 부분익절 또는 짧은 매도 후보로 본다.
          3. class_2는 기존 7% target 후보로 본다.
          4. class_3은 트레일링 스탑으로 더 끌고 갈 후보로 본다.
          
        4일 안에 고가 target_pct 이상을 한 번도 못 찍고
        저가가 stop_loss 이하로 밀리면
        데드캣 가능성이 높다.
        """

        validation_row = {
            "stop_loss": STOP_LOSS,
            "target_pct": REQUIRED_HIGH_RETURN,
            "target_class": target_class,              # class 0~3

            "day_to_4": trade_labels["day_to_4"],      # 4% 도달 날짜.. 0이면 class0
            "day_to_7": trade_labels["day_to_7"],      # 7% 도달 날짜
            "day_to_12": trade_labels["day_to_12"],    # 12% 도달 날짜
            "stop_day": trade_labels["stop_day"],      # 이탈 발생 날짜

            "target_before_stop_7": trade_labels["target_before_stop_7"],        # 이탈 전에 성공
            "stop_before_target_7": trade_labels["stop_before_target_7"],        # 성공 전에 이탈
            "target_stop_same_day_7": trade_labels["target_stop_same_day_7"],    # 이탈 성공 같은 날
            "no_target_no_stop_7": trade_labels["no_target_no_stop_7"],          # 횡보

            "target_before_stop_12": trade_labels["target_before_stop_12"],      # 이탈 전에 성공
            "stop_before_target_12": trade_labels["stop_before_target_12"],      # 성공 전에 이탈
            "target_stop_same_day_12": trade_labels["target_stop_same_day_12"],  # 이탈 성공 같은 날
            "no_target_no_stop_12": trade_labels["no_target_no_stop_12"],        # 횡보

            "fast_success_7": trade_labels["fast_success_7"],    # ~4거래일만에 달성
            "slow_success_7": trade_labels["slow_success_7"],    # 5~7거래일만에 달성
            "fail_success_7": trade_labels["fail_success_7"],    # 도달 실패

            "fast_success_12": trade_labels["fast_success_12"],  # ~4거래일만에 달성
            "slow_success_12": trade_labels["slow_success_12"],  # 5~7거래일만에 달성
            "fail_success_12": trade_labels["fail_success_12"],  # 도달 실패


            # 고가 기준 최대 수익률
            "validation_high_rate_max": validation_high_rate_max,
            "validation_high_rate_max_adj": validation_high_rate_max * EXECUTION_RATIO,

            # 저가 기준 최저 수익률
            "validation_low_rate_min": validation_low_rate_min,
        }

        rate_groups = {
            "close": close_rates,
            "high": high_rates,
            "low": low_rates,
            "open": open_rates,
        }

        # OHLC 기준 일별 수익률
        for prefix, rates in rate_groups.items():
            for i, rate in enumerate(rates, start=1):
                validation_row[f"validation_{prefix}_rate{i}"] = rate

        row.update(validation_row)


    ########################################################################

    row.update(rule_features)
    row = round_float_features(row)
    row["ticker"] = ticker  # 종목코드 숫자 float 처리 되어서 밖으로 뺌

    return {
        "row": row,
    }

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

# 최근 5일 동안 상승일 거래대금이 하락일 거래대금보다 얼마나 강했는가, _ma5_ma20_gap_chg_1d 유사, 거래대금이 작으면 설명력 부족
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

if __name__ == "__main__":
    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 7_find_low_point.py...')
    print('=== 7일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -2.5% 이상 하락이 존재, 오늘 +3.3% 이상 상승 ===')

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
                up_cnt = int(selected["target_before_stop_7"].sum())
                shortfall_cnt = total_cnt - up_cnt
                total_up_rate = up_cnt / total_cnt * 100 if total_cnt else 0

                print(f"\n룰 통과 수: {total_cnt}")
                print(f"룰 통과 후 성공률: {total_up_rate:.2f}% ({up_cnt} / {total_cnt})")

            else:
                # # 데드캣 필터 패스한 것만 룰 마이닝
                # avoid_conditions = lowscan_avoid_rules.build_conditions(df)
                #
                # avoid_mask = np.zeros(len(df), dtype=bool)
                # for cond in avoid_conditions.values():
                #     avoid_mask |= cond
                #
                # selected = df[~avoid_mask].copy()

                # CSV 저장
                df.to_csv(CSV_PATH, index=False)  # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음

                # 데드캣 필터 통과한 것만 저장
                # selected.to_csv(CSV_PATH, index=False)

                # 내림차순 정렬
                saved = sort_csv_by_today_desc(
                    in_path=CSV_PATH,
                    out_path=SORTED_CSV_PATH,
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
                    f"v1_{today} {stock_name} [{ticker}] "
                    f"일봉 차트 - 당일 상승_{row['today_pct']}%_7일 최고 수익률_{row['validation_high_rate_max_adj']}% "
                    f"rules={row['matched_rules'][:80]}"
                )

                final_file_name = (
                    f"v1_{today} {stock_name} [{ticker}] "
                    f"{row['today_pct']}%_{row['validation_high_rate_max_adj']}%.webp"
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

        print(f"7_find_low_point_processPool.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
        # log_file.close()

    except KeyboardInterrupt:
        print("\n사용자 중지 요청 감지. 작업을 종료합니다.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise