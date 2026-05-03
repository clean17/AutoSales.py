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
import lowscan_rules as rule0
modules = [rule0]

# import lowscan_rules_80_25_4_42 as rule1
# import lowscan_rules_77_25_5_42 as rule2
# modules = [rule1]

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
    sort_csv_by_today_desc, safe_read_pickle, safe_rate, to_float, round_float_features

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
output_dir = 'F:\\5below20_test'
# output_dir = 'F:\\5below20'

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7
render_graph = True

BATCH_SIZE = 20       # 작업 사이즈
START_OFFSET = 15      # 1이면 어제 기준부터 검증 가능.. 7일 검증을 사용하려면 7사용
END_OFFSET = 30      # 과거 300거래일까지 생성


def process_one(idx, count, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    # df = pd.read_pickle(filepath)
    df = safe_read_pickle(filepath)

    # 데이터가 부족하면 패스
    if df is None or df.empty or len(df) < 70:
        return None

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

    today = data.index[-1].strftime("%Y%m%d") # 마지막 인덱스
    if count == 0:
        # print('─────────────────────────────────────────────────────────────')
        print(data.index[-1].date())
        # print('─────────────────────────────────────────────────────────────')


    ########################################################################

    trading_value = data['종가'] * data['거래량']
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 마지막 20일 거래대금 평균


    # ★★★★★ x거래일 평균 거래대금 3억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # if mean_prev3 / 100_000_000 < 3:
    if mean_prev20 / 100_000_000 < 3:
        return


    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    data, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)

    # 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)

    # drop 이후 3차 생성
    data = add_technical_features(data)

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 70:
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
        "tr_volume_rank_20d": tr_volume_rank_20d,

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

    # data에 컬럼이 없거나 NaN이면 넣기 (기존 컬럼 있으면 덮어쓸지 말지는 옵션)
    data = data.copy()
    for k, v in rule_features.items():
        data[k] = v

    if "MACD_hist_3d_rank" not in data.columns:
        data["MACD_hist_3d_rank"] = np.nan

    # 여러 모듈(modules)에서 조건(rule)을 검사해서, 하나라도 만족하면 해당 모듈/룰을 선택하는 로직
    for mod in modules:
        try:
            rule_masks = mod.build_conditions(data)   # dict: rule_name -> Series[bool]
        except KeyError as e:
            print(f"[{ticker}] rule build_conditions KeyError in {mod.__name__}: {e} (missing column in data)")
            return

        RULE_NAMES = mod.RULE_NAMES

        true_conds = [
            name for name in RULE_NAMES
            if name in rule_masks and bool(rule_masks[name].iloc[-1])
        ]

        # 이 모듈에서 하나라도 True면 통과 → 다음 로직 진행
        if true_conds:
            # 필요하면 어떤 모듈/룰이었는지 저장
            matched_module = mod.__name__
            matched_rules = true_conds
            break
    else:
        # 모든 모듈을 다 봤는데도 True가 하나도 없으면 pass
        return


    ########################################################################


    row = {
        "ticker": ticker,
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "predict_str": predict_str,                      # 상승/미달

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


    origin = []
    plot_job = {}

    if render_graph:
        origin = df.copy()
        #연산하는 시간 걸리니 그래프 안그리면 패스
        # 2차 생성 feature
        origin = add_technical_features(origin)
        # 결측 제거
        o_cleaned, o_cols_to_drop = drop_sparse_columns(origin, threshold=0.10, check_inf=True, inplace=True)
        origin = o_cleaned
        # 거래정지/이상치 행 제거
        origin, o_removed_idx = drop_trading_halt_rows(origin)

        today_str = str(today)
        title = f"{today_str} {stock_name} [{ticker}] {round(data.iloc[-1]['등락률'], 2)}% Daily Chart - {predict_str} {validation_chg_rate}%"
        final_file_name = f"{today} {stock_name} [{ticker}] {round(data.iloc[-1]['등락률'], 2)}%_{predict_str}.webp"
        os.makedirs(output_dir, exist_ok=True)
        final_file_path = os.path.join(output_dir, final_file_name)

        plot_job['origin'] = origin
        plot_job['today'] = today_str
        plot_job['title'] = title
        plot_job['save_path'] = final_file_path


    return {
        "row": row,
        "plot_job": plot_job,
    }



if __name__ == "__main__":
    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 7_find_low_point.py...')
    print('=== 7일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -2.5%이 존재 + 오늘 3.3% 이상 상승 ===')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())
    # tickers = extract_numbers_from_filenames(directory = r'D:\5below20_test\4퍼', isToday=False)


    shortfall_cnt = 0    # 미달 수량
    up_cnt = 0           # 성공 수량
    rows = []            # 결과 종목 데이터 저장
    plot_jobs = []       # 그래프 생성용 데이터 저장


    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        futures = []  # 작업 저장 리스트

        # 전체 작업을 BATCH_SIZE 단위로 나눠서 반복 처리
        for batch_start in range(START_OFFSET, END_OFFSET + 1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE - 1, END_OFFSET)

            print(f"processing offset {batch_start} ~ {batch_end}")
            # cur_idx 만큼 데이터셋을 뒤에서부터 자른다 >> 잘라낸 마지막 리스트가 검증 리스트
            """
              cur_idx: 며칠 전 기준으로 검증할 것인가
              cur_idx = 1   → 오늘 기준 하루 전까지 data, 이후 7일 검증
              cur_idx = 7   → 7거래일 전까지 data, 이후 7일 검증
              cur_idx = 100 → 100거래일 전까지 data, 이후 7일 검증
              cur_idx가 7부터 올라가면서 검증 데이터셋을 만든다
            """
            for cur_idx in range(batch_start, batch_end + 1):
                # print('cur_idx', cur_idx)
                for count, ticker in enumerate(tickers):
                    # 작업 제출: process_one 함수를 병렬 실행 >> 제출된 작업을 futures 리스트에 저장
                    futures.append(executor.submit(process_one, cur_idx, count, ticker, tickers_dict))

            # 제출된 작업이 끝날 때까지 대기
            for fut in as_completed(futures):
                fut.result()   # 예외 발생 시 여기서 터져서 디버깅 쉬움

            # 다음 배치로 idx 이동
            idx = batch_end

        # 모든 작업이 완료되면 하나씩 꺼내서 집계
        for f in as_completed(futures):
            try:
                res = f.result()
            except Exception as e:
                print("worker error:", e)
                continue

            if res is None:
                continue

            row = res["row"]
            plot_job = res["plot_job"]

            rows.append(row)
            if render_graph:
                plot_jobs.append(plot_job)   # 그래프 생성하지 않으려면 주석

            if row["predict_str"] == "미달":
                shortfall_cnt += 1
            else:
                up_cnt += 1


    rows_sorted = sorted(rows, key=lambda row: row['today'])

    # 🔥 여기서 한 번에, 깔끔하게 출력
    # for row in rows_sorted:
        # print(f"\n {row['today']}   {row['stock_name']} [{row['ticker']}] {row['predict_str']}")
        # print(f"  오늘 등락률        : {row['today_pct']}%")
        # print(f"  검증 등락률(max)   : {row['validation_chg_rate']}%")
        # print(f"  검증 등락률1       : {row['validation_chg_rate1']}%")
        # print(f"  검증 등락률2       : {row['validation_chg_rate2']}%")
        # print(f"  검증 등락률3       : {row['validation_chg_rate3']}%")
        # print(f"  검증 등락률4       : {row['validation_chg_rate4']}%")
        # print(f"  검증 등락률5       : {row['validation_chg_rate5']}%")
        # print(f"  검증 등락률6       : {row['validation_chg_rate6']}%")
        # print(f"  검증 등락률7       : {row['validation_chg_rate7']}%")


    if len(rows) > 0:
        if shortfall_cnt + up_cnt == 0:
            total_up_rate = 0
        else:
            total_up_rate = up_cnt / (shortfall_cnt + up_cnt) * 100

            if render_graph is False:
                # CSV 저장
                df = pd.DataFrame(rows)

                df["MACD_hist_3d_rank"] = (
                    df.groupby("today")["MACD_hist_3d"]
                    .rank(pct=True)
                    .round(4)
                )

                df.to_csv('csv/low_result_7.csv', index=False)  # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
                saved = sort_csv_by_today_desc(
                    in_path=r"csv/low_result_7.csv",
                    out_path=r"csv/low_result_7_desc.csv",
                )
                print("saved:", saved)

        print(f"저점 매수 스크립트 결과 : {total_up_rate:.2f}% ({up_cnt} / {shortfall_cnt + up_cnt})")



    # 싱글 스레드로 그래프 처리
    for job in plot_jobs:
        # 그래프 생성
        fig = plt.figure(figsize=(14, 16), dpi=150)
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

        ax_d_price = fig.add_subplot(gs[0, 0])
        ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
        ax_w_price = fig.add_subplot(gs[2, 0])
        ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

        plot_candles_daily(job["origin"], show_months=4, title=f'{job["title"]}',
                            ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5, today=job["today"])

        plot_candles_weekly(job["origin"], show_months=12, title="Weekly Chart",
                            ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        plt.savefig(job["save_path"], format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    if len(plot_jobs) > 0:
        print('\n그래프 생성 완료')



    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
    # log_file.close()
    # print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

