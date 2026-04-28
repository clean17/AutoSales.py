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
render_graph = False

BATCH_SIZE = 20       # 작업 사이즈
START_OFFSET = 7      # 1이면 어제 기준부터 검증 가능.. 7일 검증을 사용하려면 7사용
END_OFFSET = 300       # 과거 300거래일까지 생성


def process_one(idx, count, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    # df = pd.read_pickle(filepath)
    df = safe_read_pickle(filepath)

    # 데이터가 부족하면 패스
    if df.empty or len(df) < 70:
        return

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

    trading_value = data['거래량'] * data['종가']
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
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

    # 데드캣 바운드 제거
    # if ma5_chg_rate <= 0:
    #     return

    """
    depth4 (진입 gate) >> 실패 줄이기
    - 오늘 +3% 이상
    - 최근 7일 하락 존재
    - 거래량 증가
    - 중기 위치 확인
    - 양봉 비율
    - 당일 종가 위치
    
    depth5 (점수) >> 수익률 극대화
    - find_low_scan_condition.py 스크립트로 만든 조건
    """

    # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 4% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    signal = signal_any_drop(data, 7, 3.0, -2.5)
    # signal = signal_any_drop(data, 8, 3.5, -3.0)
    if not signal:
        return


    ########################################################################
    # feature 만들기

    last = data.iloc[-1]
    last5_ret            = data['등락률'].tail(5)
    last15_ret           = data['등락률'].tail(15)
    last20_ret           = data['등락률'].tail(20)
    last60_ret           = data['등락률'].tail(60)


    # 과매도 전환 지표
    RSI14                = data['RSI14'].iloc[-1]
    RSI_rebound          = RSI14 - data['RSI14'].iloc[-3]

    # 낙폭 위험
    drawdown_60d         = last['drawdown_60d']
    dist_to_ma20         = last['dist_to_ma20']
    # 저점 반등
    rebound_from_20d_low = data['rebound_from_20d_low'].iloc[-1]
    # 저점 대비 위치 정규화
    rebound_strength = min(rebound_from_20d_low / (abs(drawdown_60d) + 1e-6), 5)

    # 갭상승 후 밀림 → 데드캣 가능성 높음
    gap_pct              = (data['시가'].iloc[-1] - data['종가'].iloc[-2]) / data['종가'].iloc[-2]

    # 거래 유입
    volume_ratio         = last['volume_ratio']

    if mean_prev3 <= 0 or not np.isfinite(mean_prev3):
        tr_value_ratio = 0
    else:
        tr_value_ratio = today_tr_val / mean_prev3
    tr_value_ratio = np.log1p(np.clip(tr_value_ratio, 0, 8.5))

    # 당일 등락률
    today_pct            = round(last['등락률'], 2)
    close_pos            = last['close_pos']

    # 상승 + 거래량 결합
    volume_price_power = today_pct * volume_ratio
    # 추세 필터 강화
    trend_filter = dist_to_ma20 * close_pos
    # 변동성 대비 상승
    raw_eff = today_pct / (tr_value_ratio + 1e-6)
    raw_eff = np.clip(raw_eff, 0, np.percentile(raw_eff, 95))
    efficiency = np.log1p(raw_eff)

    # 오늘 반등의 질이 좋은가, 종합점수
    rebound_power = max(today_pct, 0) * close_pos * tr_value_ratio

    # 많이 깨졌고, 아직 20일선 아래 깊게 있으면 위험
    deadcat_risk = abs(min(drawdown_60d, 0)) / 50 * max(0, -dist_to_ma20)

    # 반등은 강한데, 구조적 위험은 낮은 종목을 고르기 위한 점수
    raw_score = rebound_power / (1 + deadcat_risk)
    bottom_buy_score = np.clip(raw_score, 0, np.percentile(raw_score, 95))

    ############################  deadcat_filter  ###########################

    # if RSI_rebound < -15.25:
    #     return
    # if gap_pct < -0.09: # 성공 최소값
    #     return

    ########################################################################

    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)

    m_closes  = m_data['종가']
    m_max     = m_closes.max()
    m_min     = m_closes.min()
    m_current = m_closes.iloc[-1]                               # 오늘 종가, 검증 데이터로 잘랐다면 검증 직전까지의 마지막 값 (수익률 분석 용도)

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

    # --- build_conditions()가 참조하는 컬럼들을 data에 주입 (스칼라 → 컬럼 브로드캐스트) ---
    rule_features = {
        # 반등 방향
        "RSI_rebound": RSI_rebound,

        # 상승 강도
        "today_pct": today_pct,

        # 거래량
        "volume_ratio": volume_ratio,
        "tr_value_ratio": tr_value_ratio,

        # 구조 / 리스크
        "dist_to_ma20": dist_to_ma20,
        "drawdown_60d": drawdown_60d,
        "rebound_from_20d_low": rebound_from_20d_low,

        # 점수
        "bottom_buy_score": bottom_buy_score,

        "volume_price_power": volume_price_power,
        "rebound_power": rebound_power,
        "rebound_strength": rebound_strength,
        "efficiency": efficiency,
        "trend_filter": trend_filter,
    }

    # data에 컬럼이 없거나 NaN이면 넣기 (기존 컬럼 있으면 덮어쓸지 말지는 옵션)
    data = data.copy()
    for k, v in rule_features.items():
        data[k] = v


    # 여러 모듈(modules)에서 조건(rule)을 검사해서, 하나라도 만족하면 해당 모듈/룰을 선택하는 로직
    # for mod in modules:
    #     try:
    #         rule_masks = mod.build_conditions(data)   # dict: rule_name -> Series[bool]
    #     except KeyError as e:
    #         print(f"[{ticker}] rule build_conditions KeyError in {mod.__name__}: {e} (missing column in data)")
    #         return
    #
    #     RULE_NAMES = mod.RULE_NAMES
    #
    #     true_conds = [
    #         name for name in RULE_NAMES
    #         if name in rule_masks and bool(rule_masks[name].iloc[-1])
    #     ]
    #
    #     # 이 모듈에서 하나라도 True면 통과 → 다음 로직 진행
    #     if true_conds:
    #         # 필요하면 어떤 모듈/룰이었는지 저장
    #         matched_module = mod.__name__
    #         matched_rules = true_conds
    #         break
    # else:
    #     # 모든 모듈을 다 봤는데도 True가 하나도 없으면 pass
    #     return


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
    print(' x일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -x%이 존재 + 오늘 x% 이상 상승')

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


    print('shortfall_cnt', shortfall_cnt)
    print('up_cnt', up_cnt)
    if shortfall_cnt+up_cnt==0:
        total_up_rate=0
    else:
        total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100

        # CSV 저장
        pd.DataFrame(rows).to_csv('csv/low_result_7.csv', index=False) # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
        saved = sort_csv_by_today_desc(
            in_path=r"csv/low_result_7.csv",
            out_path=r"csv/low_result_7_desc.csv",
        )
        print("saved:", saved)

    print(f"저점 매수 스크립트 결과 : {total_up_rate:.2f}%")



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

