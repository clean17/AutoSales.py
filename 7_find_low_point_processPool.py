'''
저점을 찾는 스크립트
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승
3일 평균 거래대금이 1000억 이상이면 무조건 사야한다
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
import lowscan_rules_77_25_5_42 as rule0
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
    sort_csv_by_today_desc, safe_read_pickle

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
output_dir = 'D:\\5below20_test'
# output_dir = 'D:\\5below20'

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 8
render_graph = False


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

    # idx만큼 뒤에서 자른다 (idx가 2라면 2일 전 데이터셋)
    if idx != 0:
        data = df[:-idx]
        remaining_data = df[len(df)-idx:]
    else:
        data = df
        remaining_data = None

    if data.empty:
        return None

    today = data.index[-1].strftime("%Y%m%d") # 마지막 인덱스
    if count == 0:
        # print('─────────────────────────────────────────────────────────────')
        print(data.index[-1].date())
        # print('─────────────────────────────────────────────────────────────')


    ########################################################################

    trading_value = data['거래량'] * data['종가']


    # 직전 날까지의 마지막 3일 거래대금 평균
    today_tr_val = trading_value.iloc[-1]
    mean_prev3 = trading_value.iloc[:-1].tail(3).mean()

    # ★★★★★ 3거래일 평균 거래대금 5억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if round(mean_prev3, 1) / 100_000_000 < 3:
        return


    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    data = cleaned

    # 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)

    # 5일, 20일 이동평균선 없으면 패스
    if 'MA5' not in data.columns or 'MA20' not in data.columns:
        return

    # 마지막 일자 5일선은 20일선보다 낮아야 한다
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    ma5_chg_rate = (ma5_today - ma5_yesterday) / ma5_yesterday * 100


    # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 4% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # signal = signal_any_drop(data, 12, 4.0 ,-3.0) # 40/55 ---
    # signal = signal_any_drop(data, 10, 4.0 ,-2.0) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.2) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.6) # 48/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.8) # 46/78
    signal = signal_any_drop(data, 7, 3.0, -2.5) # 45/71 ---
    # signal = signal_any_drop(data, 10, 4.0 ,-3.2) # 44/68
    # signal = signal_any_drop(data, 10, 4.0 ,-3.4) # 42/64
    # signal = signal_any_drop(data, 10, 4.0 ,-3.6) # 39/57
    # signal = signal_any_drop(data, 10, 4.0 ,-3.8) # 37/49 ---
    # signal = signal_any_drop(data, 10, 4.0 ,-4.0) # 34/44
    # signal = signal_any_drop(data, 10, 4.0 ,-2.5) # 49/83
    # signal = signal_any_drop(data, 9, 4.0 ,-2.5) # 50/85
    # signal = signal_any_drop(data, 8, 4.0 ,-2.5) # 46/92
    # signal = signal_any_drop(data, 7, 4.0 ,-2.5) # 46/92
    # signal = signal_any_drop(data, 6, 4.0 ,-2.5) # 40/92
    if not signal:
        return


    ########################################################################

    # ★★★★★ 최근 20일 변동성 너무 낮으면 제외 (지루한 종목)
    last15_ret = data['등락률'].tail(15)           # 등락률이 % 단위라고 가정
    last20_ret = data['등락률'].tail(20)           # 등락률이 % 단위라고 가정
    last30_ret = data['등락률'].tail(30)
    vol15 = last15_ret.std()                      # 표준편차
    vol30 = last30_ret.std()                      # 표준편차

    # 양봉 비율이 30% 미만이면 제외 (계속 음봉 위주)
    pos20_ratio = (last20_ret > 0).mean()           # True 비율 => 양봉 비율

    # 추가 독립 피쳐
    def to_float(x):
        return float(x) if pd.notna(x) else np.nan

    last = data.iloc[-1]
    close_pos        = round(to_float(last.get("close_pos")), 4)

    ########################################################################

    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)

    m_closes = m_data['종가']
    m_max = m_closes.max()
    m_min = m_closes.min()
    m_current = m_closes[-1]

    if remaining_data is not None:
        r_data = remaining_data[:7]   # 10 > 7거래일로 수정
        # r_closes = r_data['종가']
        r_closes = remaining_data['종가'].iloc[:7].reset_index(drop=True)
        r_closes = r_closes.reindex(range(7))  # 0~6 없으면 NaN으로 채움

        # r_max = r_closes.max()
        r_max = r_closes.max(skipna=True)

        r1, r2, r3, r4, r5, r6, r7 = (r_closes.iloc[i] for i in range(7))

        def safe_rate(x, base):
            if pd.isna(x) or base == 0 or not np.isfinite(base):
                return np.nan
            return (x - base) / base * 100

        # validation_chg_rate = (r_max-m_current)/m_current*100    # 검증 등락률
        validation_chg_rate  = safe_rate(r_max, m_current)
        validation_chg_rate1 = safe_rate(r1, m_current)
        validation_chg_rate2 = safe_rate(r2, m_current)
        validation_chg_rate3 = safe_rate(r3, m_current)
        validation_chg_rate4 = safe_rate(r4, m_current)
        validation_chg_rate5 = safe_rate(r5, m_current)
        validation_chg_rate6 = safe_rate(r6, m_current)
        validation_chg_rate7 = safe_rate(r7, m_current)

    else:
        validation_chg_rate = 0

    three_m_chg_rate=(m_max-m_min)/m_min*100        # 최근 3개월 동안의 등락률
    today_chg_rate=(m_current-m_max)/m_max*100      # 최근 3개월 최고 대비 오늘 등락률 계산



    result = low_weekly_check(m_data)
    if result["ok"]:
        # ★★★★★ 저번주 대비 이번주 증감률 -1%보다 낮으면 패스 (아직 하락 추세) ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if result["is_drop_more_than_minus1pct"]:
            # return
            pass


    ########################################################################

    ma5_chg_rate = round(ma5_chg_rate, 4)
    vol15 = round(vol15, 4)
    vol30 = round(vol30, 4)
    pos20_ratio = round(pos20_ratio*100, 4)
    mean_prev3 = round(mean_prev3, 4)
    today_tr_val = round(today_tr_val, 4)
    three_m_chg_rate = round(three_m_chg_rate, 4)
    today_chg_rate = round(today_chg_rate, 4)
    pct_vs_lastweek = round(result['pct_vs_lastweek'], 4)
    pct_vs_last4week = round(result['pct_vs_last4week'], 4)
    today_pct = round(data.iloc[-1]['등락률'], 2)
    validation_chg_rate = round(validation_chg_rate, 2)
    validation_chg_rate1 = round(validation_chg_rate1, 2)
    validation_chg_rate2 = round(validation_chg_rate2, 2)
    validation_chg_rate3 = round(validation_chg_rate3, 2)
    validation_chg_rate4 = round(validation_chg_rate4, 2)
    validation_chg_rate5 = round(validation_chg_rate5, 2)
    validation_chg_rate6 = round(validation_chg_rate6, 2)
    validation_chg_rate7 = round(validation_chg_rate7, 2)

    predict_str = '상승'
    if validation_chg_rate < VALIDATION_TARGET_RETURN:
        predict_str = '미달'


    # --- build_conditions()가 참조하는 컬럼들을 data에 주입 (스칼라 → 컬럼 브로드캐스트) ---
    rule_features = {
        "ma5_chg_rate": ma5_chg_rate,                    # 5일선 기울기 👍
        "vol15": vol15,                                  # 20일 평균 변동성
        "vol30": vol30,                                  # 30일 평균 변동성
        "pos20_ratio": pos20_ratio,                      # 20일 평균 양봉비율 (전환 직전 눌림/반등 준비를 더 잘 반영할 가능성)
        "today_tr_val": today_tr_val,                    # 오늘 거래대금 👍
        "mean_prev3": mean_prev3,                        # 직전 3일 평균 거래대금 (조건에서 다수 사용)
        "three_m_chg_rate": three_m_chg_rate,            # 3개월 종가 최저 대비 최고 등락률 👍
        "today_chg_rate": today_chg_rate,                # 3개월 종가 최고 대비 오늘 등락률 👍
        "pct_vs_lastweek": pct_vs_lastweek,              # 저번주 대비 이번주 등락률
        "pct_vs_last4week": pct_vs_last4week,            # 4주 전 대비 이번주 등락률
        "today_pct": today_pct,                          # 오늘등락률 👍
        "close_pos": close_pos,                          # 당일 range 내 종가 위치(0~1)
    }

    # data에 컬럼이 없거나 NaN이면 넣기 (기존 컬럼 있으면 덮어쓸지 말지는 옵션)
    data = data.copy()
    for k, v in rule_features.items():
        data[k] = v


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

    """
    높은 전환관계의 피쳐들
    close_pos, today_pct, ma5_chg_rate
    """
    row = {
        "ticker": ticker,
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "predict_str": predict_str,                      # 상승/미달

        "ma5_chg_rate": ma5_chg_rate,                    # 5일선 기울기 👍
        "vol15": vol15,                                  # 15일 평균 변동성
        "vol30": vol30,                                  # 30일 평균 변동성
        "pos20_ratio": pos20_ratio,                      # 20일 평균 양봉비율 (전환 직전 눌림/반등 준비를 더 잘 반영할 가능성)

        "mean_prev3": mean_prev3,                        # 직전 3일 평균 거래대금 (조건에서 다수 사용)
        "today_tr_val": today_tr_val,                    # 오늘 거래대금 👍

        "three_m_chg_rate": three_m_chg_rate,            # 3개월 종가 최저 대비 최고 등락률 👍
        "today_chg_rate": today_chg_rate,                # 3개월 종가 최고 대비 오늘 등락률 👍
        "pct_vs_lastweek": pct_vs_lastweek,              # 저번주 대비 이번주 등락률
        "pct_vs_last4week": pct_vs_last4week,            # 4주 전 대비 이번주 등락률
        "today_pct": today_pct,                          # 오늘등락률 👍

        "close_pos": close_pos,                          # 당일 range 내 종가 위치(0~1)


        "validation_chg_rate": validation_chg_rate,      # 검증 등락률
        "validation_chg_rate1": validation_chg_rate1,    # 검증 등락률
        "validation_chg_rate2": validation_chg_rate2,    # 검증 등락률
        "validation_chg_rate3": validation_chg_rate3,    # 검증 등락률
        "validation_chg_rate4": validation_chg_rate4,    # 검증 등락률
        "validation_chg_rate5": validation_chg_rate5,    # 검증 등락률
        "validation_chg_rate6": validation_chg_rate6,    # 검증 등락률
        "validation_chg_rate7": validation_chg_rate7,    # 검증 등락률
    }


    origin = df.copy()

    if render_graph:
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

    # 그래프 그릴 때 필요한 것만 모아서 리턴
    plot_job = {
        "origin": origin,
        "today": today_str,
        "title": title,
        "save_path": final_file_path,
    }


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

    shortfall_cnt = 0
    up_cnt = 0
    rows=[]
    plot_jobs = []

    # 10이면, 10거래일의 하루전부터, -1이면 어제
    # origin_idx = idx = -1
    origin_idx = idx = 9
    workers = os.cpu_count()
    BATCH_SIZE = 20

    # end_idx = origin_idx + 170 # 마지막 idx (05/13부터 데이터 만드는 용)
    end_idx = origin_idx + 50 # 마지막 idx
    # end_idx = origin_idx + 1 # 그날 하루만

    with ProcessPoolExecutor(max_workers=workers - 2) as executor:
        futures = []

        while idx < end_idx:
            batch_end = min(idx + BATCH_SIZE, end_idx)

            # idx를 배치 단위로 1씩 증가시키며(최대 10번) 작업 제출
            for cur_idx in range(idx + 1, batch_end + 1):
                # print('cur_idx', cur_idx)
                for count, ticker in enumerate(tickers):
                    futures.append(executor.submit(process_one, cur_idx, count, ticker, tickers_dict))

            # 이번 배치가 끝날 때까지 대기
            for fut in as_completed(futures):
                fut.result()   # 예외 발생 시 여기서 터져서 디버깅 쉬움

            # 다음 배치로 idx 이동
            idx = batch_end

        # 완료된 것부터 하나씩 받아서 집계
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
    for row in rows_sorted:
        print(f"\n {row['today']}   {row['stock_name']} [{row['ticker']}] {row['predict_str']}")
        # print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        # print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  오늘 등락률        : {row['today_pct']}%")
        print(f"  검증 등락률(max)   : {row['validation_chg_rate']}%")
        print(f"  검증 등락률1       : {row['validation_chg_rate1']}%")
        print(f"  검증 등락률2       : {row['validation_chg_rate2']}%")
        print(f"  검증 등락률3       : {row['validation_chg_rate3']}%")
        print(f"  검증 등락률4       : {row['validation_chg_rate4']}%")
        print(f"  검증 등락률5       : {row['validation_chg_rate5']}%")
        print(f"  검증 등락률6       : {row['validation_chg_rate6']}%")
        print(f"  검증 등락률7       : {row['validation_chg_rate7']}%")


    print('shortfall_cnt', shortfall_cnt)
    print('up_cnt', up_cnt)
    if shortfall_cnt+up_cnt==0:
        total_up_rate=0
    else:
        total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100

        # CSV 저장
        # pd.DataFrame(rows).to_csv('csv/low_result_7.csv', index=False) # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
        # saved = sort_csv_by_today_desc(
        #     in_path=r"csv/low_result_7.csv",
        #     out_path=r"csv/low_result_7_desc.csv",
        # )
        # print("saved:", saved)

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
    print('\n그래프 생성 완료')

    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
    # log_file.close()
    # print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

