'''
저점을 찾는 스크립트 (저점매수 + 반등 + 모멘텀)
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승
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
render_graph = False

START_OFFSET = 7      # 1이면 어제 기준부터 검증 가능.. 7일 검증을 사용하려면 7사용
END_OFFSET = 300      # 과거 300거래일까지 생성

def process_ticker(ticker, tickers_dict, i):
    results = []

    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[process_ticker] {stock_name} ({ticker}) 파일 없음")
        return results

    df = safe_read_pickle(filepath)

    if render_graph is False:
        df = df[:-100]  # 검증 데이터 분리 테스트

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
        return

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

    ########################################################################

    closes = data['종가'].values
    trading_value = data['trading_value']
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 마지막 20일 거래대금 평균


    # ★★★★★ 20거래일 평균 거래대금 3억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if mean_prev20 / 100_000_000 < 3:
        return

    # 5일, 20일 이동평균선 없으면 패스
    REQUIRED_COLS = ["MA5", "MA20", "등락률"]

    for col in REQUIRED_COLS:
        if col not in data.columns:
            return

    # 오늘의 5일션 변동율 계산 (퍼센트로 보려면 * 100)
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]
    ma5_chg_rate = round(safe_rate(ma5_today, ma5_yesterday), 3)

    """
    depth4 (진입 gate) >> 실패 줄이기
    - 오늘 +3% 이상
    - 최근 7일 하락 존재
    - 거래량 증가
    
    depth5 (점수) >> 수익률 극대화
    - find_low_scan_condition.py 스크립트로 만든 조건
    """

    # 오늘 제외 최근 7일 5일선이 20일선보다 계속 낮은데 -2.5% 하락이 있으면서 오늘 3.3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    signal = signal_any_drop(data, 7, 3.3, -2.5)
    # signal = signal_any_drop(data, 7, 2, -2.5)
    # signal = signal_any_drop(data, 8, 3.5, -3.0)
    if not signal:
        return

    ########################################################################
    # feature 만들기

    last = data.iloc[-1]

    _ma5_slope_3d       = (ma5_today - data['MA5'].iloc[-4]) / 3
    _ma20_slope_3d      = (data['MA20'].iloc[-1] - data['MA20'].iloc[-4]) / 3
    trend_signal        = _ma5_slope_3d - _ma20_slope_3d * 0.5
    # trend_signal_tanh   = np.tanh(trend_signal / 50)

    # 최근 7거래일 최대 하락 (오늘 제외, 어제 포함)
    _recent_7_ret       = data['등락률'].iloc[-8:-1]
    max_drop_7d         = _recent_7_ret.min()
    # 하락 일수
    # neg_days_7d       = (_recent_7_ret < 0).sum()

    # -------------------------------
    # 내부 계산값
    # -------------------------------
    # 하루 변화량 : 빠름, 노이즈 많음, 3일 : 신호는 늦엇지만 안정된 필터용
    _MACD_hist_1d       = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-2]
    MACD_hist_3d        = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-4]
    MACD_acc            = _MACD_hist_1d - (MACD_hist_3d / 3)
    # 최소 변별력이 없음
    # MACD_rebound_power = (
    #         np.tanh(MACD_acc / 50) * 0.65 +
    #         np.tanh(MACD_hist_3d / 100) * 0.35
    # )

    # 오늘 거래대금 변동률 - 내부 계산용
    if mean_prev3 <= 0 or not np.isfinite(mean_prev3) or mean_prev5 <= 0 or not np.isfinite(mean_prev5):
        tr_value_ratio = 0
    else:
        tr_value_ratio = (today_tr_val / mean_prev3) * 0.4 + (today_tr_val / mean_prev5) * 0.6

    # 거래대금
    # tr_value_ratio_tanh  = np.tanh(np.log1p(tr_value_ratio) / 2)

    tr_values_20d = trading_value.iloc[-20:]
    today_tr_val = tr_values_20d.iloc[-1]
    tr_volume_rank_20d = (tr_values_20d <= today_tr_val).mean()

    # 한달 대비 오늘 거래량.. 변별력 없음
    # _volume_ratio        = last['volume_ratio']


    # 20일 최저점 발생일 대비 몇 일 지났는지.. 변별력이 없음 (19: 처음, 0: 오늘)
    # window               = data.iloc[-20:]
    # low_idx_pos          = window['저가'].values.argmin()   # 최솟값의 인덱스를 반환
    # days_since_low       = len(window) - 1 - low_idx_pos

    # 20일 최저점 대비 몇 % 올라왔는지
    low_20d              = data['저가'].iloc[-20:].min()
    dist_from_low        = safe_rate(last['종가'], low_20d)
    # dist_from_low_tanh   = np.tanh(dist_from_low / 20)

    today_pct            = round(last['등락률'], 2)

    # 이미 많이 오른 종목 제거.. 변별력 없음
    # _recent_runup        = data['등락률'].iloc[-5:-1].sum()

    # 데드캣 패널티.. 변별력 없음
    # _drawdown_60d        = last['drawdown_60d']
    # _dist_to_ma20        = last['dist_to_ma20']
    # _deadcat_penalty     = max(0, (-_drawdown_60d - 40) / 20) + max(0, -_dist_to_ma20 / 5)

    """
    rule_features 설명
    
    today_pct
    - 마지막 날 등락률
    - 이미 +3.3% 이상 반등한 종목 중에서도 당일 반등 강도를 보기 위한 값
    
    trend_signal
    - 5일선 단기 기울기와 20일선 중기 기울기를 합친 추세 전환 신호
    - 값이 클수록 단기 반등 힘이 강하고, 저점 반등 초입 가능성이 높음
    
    MACD_acc
    - MACD histogram의 가속도
    - 단순 상승이 아니라 반등 힘이 빨라지는지를 보기 위한 값
    - 저점에서 막 튀기 시작하는 순간 포착용
    
    MACD_hist_3d
    - 최근 3거래일 MACD histogram 변화량
    - 하루짜리 노이즈보다 안정적인 단기 모멘텀 확인용
    - MACD_acc가 타이밍이면, MACD_hist_3d는 추세 확인 역할
    
    dist_from_low
    - 최근 20일 저가 최저점 대비 현재 종가가 얼마나 올라왔는지
    - 너무 낮으면 아직 반등 확인 전, 너무 높으면 이미 늦은 구간일 수 있음
    - 저점 반등 초입 위치 판단용
    
    tr_value_ratio
    - 오늘 거래대금이 최근 3일/5일 평균 대비 얼마나 증가했는지
    - 단기 수급 유입 강도 확인용
    
    tr_volume_rank_20d
    - 오늘 거래대금이 최근 20거래일 중 어느 정도 위치인지
    - 오늘 수급이 최근 기준으로 특별한 날인지 확인하는 값
    
    max_drop_7d
    - 오늘 제외 최근 7거래일 중 가장 큰 하락률
    - 눌림 강도 확인용
    - 값이 더 작을수록, 예: -3, -5, 더 강한 눌림 후 반등
    """
    rule_features = {
        "today_pct": today_pct,

        "trend_signal": trend_signal,

        "MACD_acc": MACD_acc,
        "MACD_hist_3d": MACD_hist_3d,

        "dist_from_low": dist_from_low,

        "tr_value_ratio": tr_value_ratio,
        # "tr_volume_rank_20d": tr_volume_rank_20d,

        "max_drop_7d": max_drop_7d,
    }


    ############################  deadcat_filter  ###########################

    _gap_pct              = (data['시가'].iloc[-1] - data['종가'].iloc[-2]) / data['종가'].iloc[-2]
    _vol_ratio_15_60      = round(data['등락률'].tail(15).std() / (data['등락률'].tail(60).std() + 1e-9), 3)   # 단기 변동성과 장기 변동성을 비교하는 비율
    _RSI_rebound          = data['RSI14'].iloc[-1] - data['RSI14'].iloc[-3]
    _close_pos            = last['close_pos']
    _rebound_power        = today_pct * (0.5 + _close_pos)

    other_rule_features = {
        "_gap_pct": _gap_pct,
        "_vol_ratio_15_60": _vol_ratio_15_60,
        "_RSI_rebound": _RSI_rebound,
        "_rebound_power": _rebound_power,
        "_MACD_hist_1d": _MACD_hist_1d,
    }

    rule_features.update(other_rule_features)

    if _gap_pct < -0.092:
        return
    if _vol_ratio_15_60 < 0.246:
        return
    if _RSI_rebound < -15.247:
        return
    if _rebound_power < 1.823:
        return
    if _MACD_hist_1d < -553.4:
        return
    if MACD_acc < -479.478:
        return

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

    # 검증 데이터 (마지막 n일)
    if remaining_data is not None:
        r_closes = remaining_data['종가'].iloc[:7].reset_index(drop=True)  # Series 인덱스 새로
        r_closes = r_closes.reindex(range(7))      # 0~6 없으면 NaN으로 채움
        r_max = r_closes.max(skipna=True)          # 결측치(NaN)를 무시하고 계산

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
                    continue


                for res in results:
                    row = res["row"]
                    rows.append(row)


        rows_sorted = sorted(rows, key=lambda row: row['today'])


        if len(rows) > 0:
            df = pd.DataFrame(rows)

            df["MACD_hist_3d_rank"] = (
                df.groupby("today")["MACD_hist_3d"]
                .rank(pct=True)
                .round(4)
            )

            if render_graph:
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

            if render_graph is False:

                # CSV 저장
                df.to_csv('csv/low_result_7.csv', index=False)  # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
                saved = sort_csv_by_today_desc(
                    in_path=r"csv/low_result_7.csv",
                    out_path=r"csv/low_result_7_desc.csv",
                )
                print("saved:", saved)


        if render_graph:
            plot_jobs = []

            for _, row in selected.iterrows():
                ticker = row["ticker"]
                stock_name = row["stock_name"]
                idx = int(row["idx"])

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

                if idx != 0:
                    chart_data = origin[:-idx].copy()
                else:
                    chart_data = origin.copy()

                if chart_data.empty or len(chart_data) < 70:
                    continue

                today = chart_data.index[-1].strftime("%Y%m%d")

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


            if len(selected) > 0 and render_graph:
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