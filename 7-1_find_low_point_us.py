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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from lowscan_rules_us import build_conditions, RULE_NAMES

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import _col, get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, low_weekly_check, get_nasdaq_symbols, \
    get_usd_krw_rate, add_today_change_rate, sort_csv_by_today_desc, safe_read_pickle

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle_us')
output_dir = 'F:\\5below20_test'
# output_dir = 'F:\\5below20'

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7
render_graph = False



def process_one(idx, count, ticker, exchangeRate):
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    # print('filepath', filepath)
    # df = pd.read_pickle(filepath)
    df = safe_read_pickle(filepath)
    # print(df)

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

    closes = data['Close'].values
    trading_value = data['Volume'] * data['Close']


    # 직전 날까지의 마지막 3일 거래대금 평균
    today_tr_val = trading_value.iloc[-1]
    mean_prev3 = trading_value.iloc[:-1].tail(3).mean()
    if not np.isfinite(mean_prev3) or mean_prev3 == 0:
        chg_tr_val = 0.0
    else:
        chg_tr_val = (today_tr_val-mean_prev3)/mean_prev3*100

    # ★★★★★ 3거래일 평균 거래대금 5억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if round(mean_prev3, 1) * exchangeRate / 100_000_000 < 3:
        return


    # 2차 생성 feature
    data = add_technical_features(data)
    # 등락률 추가
    data = add_today_change_rate(data)

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
    ma20_today = data['MA20'].iloc[-1]
    ma20_yesterday = data['MA20'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    ma5_chg_rate = (ma5_today - ma5_yesterday) / ma5_yesterday * 100
    ma20_chg_rate = (ma20_today - ma20_yesterday) / ma20_yesterday * 100


    # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 4% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # signal = signal_any_drop(data, 12, 4.0 ,-3.0) # 40/55 ---
    # signal = signal_any_drop(data, 10, 4.0 ,-2.0) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.2) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.6) # 48/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.8) # 46/78
    signal = signal_any_drop(data, 7, 3.0, -2.5, 'today_chg_rate') # 45/71 ---
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
    last20_ret = data['today_chg_rate'].tail(20)           # 등락률이 % 단위라고 가정
    last30_ret = data['today_chg_rate'].tail(30)
    vol20 = last20_ret.std()                      # 표준편차
    vol30 = last30_ret.std()                      # 표준편차

    # 평균 등락률
    mean_ret20 = last20_ret.mean()
    mean_ret30 = last30_ret.mean()

    # 양봉 비율이 30% 미만이면 제외 (계속 음봉 위주)
    pos20_ratio = (last20_ret > 0).mean()           # True 비율 => 양봉 비율
    pos30_ratio = (last30_ret > 0).mean()           # True 비율 => 양봉 비율

    # 추가 독립 피쳐
    def to_float(x):
        return float(x) if pd.notna(x) else np.nan

    def to_int01(x):
        return int(bool(x)) if pd.notna(x) else 0

    last = data.iloc[-1]
    lower_wick_ratio = to_float(last.get("lower_wick_ratio"))
    close_pos        = to_float(last.get("close_pos"))
    bb_recover       = to_int01(last.get("bb_recover"))
    z20              = to_float(last.get("z20"))
    macd_hist_chg    = to_float(last.get("macd_hist_chg"))

    ########################################################################

    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)

    m_closes = m_data['Close']
    m_max = m_closes.max()
    m_min = m_closes.min()
    m_current = m_closes[-1]

    if remaining_data is not None:
        r_data = remaining_data[:7]   # 10 > 7거래일로 수정
        # r_closes = r_data['Close']
        r_closes = remaining_data['Close'].iloc[:7].reset_index(drop=True)
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
    ma20_chg_rate = round(ma20_chg_rate, 4)
    vol20 = round(vol20, 4)
    vol30 = round(vol30, 4)
    mean_ret20 = round(mean_ret20, 4)
    mean_ret30 = round(mean_ret30, 4)
    pos20_ratio = round(pos20_ratio*100, 4)
    pos30_ratio = round(pos30_ratio*100, 4)
    mean_prev3 = round(mean_prev3, 4)
    today_tr_val = round(today_tr_val, 4)
    chg_tr_val = round(chg_tr_val, 4)
    three_m_chg_rate = round(three_m_chg_rate, 4)
    today_chg_rate = round(today_chg_rate, 4)
    pct_vs_firstweek = round(result['pct_vs_firstweek'], 4)
    pct_vs_lastweek = round(result['pct_vs_lastweek'], 4)
    pct_vs_last2week = round(result['pct_vs_last2week'], 4)
    pct_vs_last3week = round(result['pct_vs_last3week'], 4)
    pct_vs_last4week = round(result['pct_vs_last4week'], 4)
    today_pct = round(data.iloc[-1]['today_chg_rate'], 4)
    validation_chg_rate = round(validation_chg_rate, 4)
    validation_chg_rate1 = round(validation_chg_rate1, 4)
    validation_chg_rate2 = round(validation_chg_rate2, 4)
    validation_chg_rate3 = round(validation_chg_rate3, 4)
    validation_chg_rate4 = round(validation_chg_rate4, 4)
    validation_chg_rate5 = round(validation_chg_rate5, 4)
    validation_chg_rate6 = round(validation_chg_rate6, 4)
    validation_chg_rate7 = round(validation_chg_rate7, 4)

    predict_str = '상승'
    if validation_chg_rate < VALIDATION_TARGET_RETURN:
        predict_str = '미달'


    # # --- build_conditions()가 참조하는 컬럼들을 data에 주입 (스칼라 → 컬럼 브로드캐스트) ---
    # rule_features = {
    #     "ma5_chg_rate": ma5_chg_rate,
    #     "ma20_chg_rate": ma20_chg_rate,
    #     "vol20": vol20,
    #     "vol30": vol30,
    #     "mean_ret20": mean_ret20,
    #     "mean_ret30": mean_ret30,
    #     "pos20_ratio": pos20_ratio,
    #     "pos30_ratio": pos30_ratio,
    #     "mean_prev3": mean_prev3,
    #     "today_tr_val": today_tr_val,
    #     "chg_tr_val": chg_tr_val,
    #     "three_m_chg_rate": three_m_chg_rate,
    #     "today_chg_rate": today_chg_rate,
    #     "pct_vs_firstweek": pct_vs_firstweek,
    #     "pct_vs_lastweek": pct_vs_lastweek,
    #     "pct_vs_last2week": pct_vs_last2week,
    #     "pct_vs_last3week": pct_vs_last3week,
    #     "pct_vs_last4week": pct_vs_last4week,
    #     "today_pct": today_pct,
    # }
    #
    # # data에 컬럼이 없거나 NaN이면 넣기 (기존 컬럼 있으면 덮어쓸지 말지는 옵션)
    # data = data.copy()
    # for k, v in rule_features.items():
    #     data[k] = v
    #
    #
    # # 룰 마스크 생성 (각 룰마다 Series[bool] 반환)
    # try:
    #     rule_masks = build_conditions(data)
    # except KeyError as e:
    #     print(f"[{ticker}] rule build_conditions KeyError: {e} (missing column in data)")
    #     return
    #
    # # 오늘(마지막 행)에서 True인 룰 이름만 추출
    # true_conds = [
    #     name for name in RULE_NAMES
    #     if name in rule_masks and bool(rule_masks[name].iloc[-1])
    # ]
    #
    # # True가 하나도 없으면 pass
    # if not true_conds:
    #     return


    ########################################################################

    row = {
        "ticker": ticker,
        "today" : str(data.index[-1].date()),
        # "3_months_ago": str(m_data.index[0].date()),   # 3달전 날짜
        "predict_str": predict_str,                      # 상승/미달

        "ma5_chg_rate": ma5_chg_rate,                    # 5일선 기울기 👍
        "vol20": vol20,                                  # 20일 평균 변동성
        "pos20_ratio": pos20_ratio,                      # 20일 평균 양봉비율 (전환 직전 눌림/반등 준비를 더 잘 반영할 가능성)
        "today_tr_val": today_tr_val,                    # 오늘 거래대금 👍
        "chg_tr_val": chg_tr_val,                        # 거래대금 변동률 (chg_tr_val이 이미 mean_prev3 대비 변화율을 담고있다)

        "three_m_chg_rate": three_m_chg_rate,            # 3개월 종가 최저 대비 최고 등락률 👍
        "today_chg_rate": today_chg_rate,                # 3개월 종가 최고 대비 오늘 등락률 👍
        "pct_vs_lastweek": pct_vs_lastweek,              # 저번주 대비 이번주 등락률
        "pct_vs_last4week": pct_vs_last4week,            # 4주 전 대비 이번주 등락률
        "today_pct": today_pct,                          # 오늘등락률 👍

        "lower_wick_ratio": lower_wick_ratio,            # 아래꼬리 비율
        "close_pos": close_pos,                          # 당일 range 내 종가 위치(0~1)
        "bb_recover": bb_recover,                        # 하단밴드 복귀 이벤트
        "z20": z20,                                      # z-score
        "macd_hist_chg": macd_hist_chg,                  # MACD hist 가속

        "validation_chg_rate": validation_chg_rate,      # 검증 등락률
        "validation_chg_rate1": validation_chg_rate1,    # 검증 등락률
        "validation_chg_rate2": validation_chg_rate2,    # 검증 등락률
        "validation_chg_rate3": validation_chg_rate3,    # 검증 등락률
        "validation_chg_rate4": validation_chg_rate4,    # 검증 등락률
        "validation_chg_rate5": validation_chg_rate5,    # 검증 등락률
        "validation_chg_rate6": validation_chg_rate6,    # 검증 등락률
        "validation_chg_rate7": validation_chg_rate7,    # 검증 등락률
        # "vol30": vol30,                                  # 30일 평균 변동성 (vol20과 중복, 7일 내 수익 목표라면 20을 사용해)
        # "pos30_ratio": pos30_ratio,                      # 30일 평균 양봉비율 (한 달 분위기(추세가 이미 시작됐는지) → 보조)
        # "mean_ret30": mean_ret30,                        # 30일 평균 등락률 (한 달 반 분위기라서 컨텍스트(보조) 성격이 강함)
        # "pct_vs_firstweek": pct_vs_firstweek,            # 3개월 주봉 첫주 대비 이번주 등락률
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
    title = f"{today_str} [{ticker}] {round(data.iloc[-1]['today_chg_rate'], 2)}% Daily Chart - {predict_str} {validation_chg_rate}%"
    final_file_name = f"{today} [{ticker}] {round(data.iloc[-1]['today_chg_rate'], 2)}%_{predict_str}.webp"
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
    print(f'{nowTime} - 🕒 running 7-1_find_low_point.py...')
    print(' 10일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승')

    exchangeRate = get_usd_krw_rate()
    if exchangeRate is None:
        print('#######################   exchangeRate is None   #######################')
    else:
        print(f'#######################   exchangeRate is {exchangeRate}   #######################')

    tickers = get_nasdaq_symbols()

    shortfall_cnt = 0
    up_cnt = 0
    rows=[]
    plot_jobs = []

    # 10이면, 10거래일의 하루전부터, -1이면 어제
    # origin_idx = idx = -1
    origin_idx = idx = 7
    workers = os.cpu_count()
    BATCH_SIZE = 20

    end_idx = origin_idx + 170 # 마지막 idx (04/15부터 데이터 만드는 용)
    # end_idx = origin_idx + 60 # 마지막 idx
    # end_idx = origin_idx + 1 # 그날 하루만

    with ProcessPoolExecutor(max_workers=workers - 2) as executor:
        futures = []

        while idx < end_idx:
            batch_end = min(idx + BATCH_SIZE, end_idx)

            # idx를 배치 단위로 1씩 증가시키며(최대 10번) 작업 제출
            for cur_idx in range(idx + 1, batch_end + 1):
                # print('cur_idx', cur_idx)
                for count, ticker in enumerate(tickers):
                    futures.append(executor.submit(process_one, cur_idx, count, ticker, exchangeRate))

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


    # 🔥 여기서 한 번에, 깔끔하게 출력
    for row in rows:
        print(f"\n {row['today']} [{row['ticker']}] {row['predict_str']}")
        # print(f"  3개월 전 날짜           : {row['3_months_ago']}")
        # print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        # print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        # print(f"  거래대금 변동률         : {row['chg_tr_val']}%")
        # # print(f"  20일선 기울기                      ( > -1.7): {row['ma20_chg_rate']}")
        # print(f"  최근 20일 변동성                   ( > 1.5%): {row['vol20']}%")
        # print(f"  최근 20일 평균 등락률              ( >= -3%): {row['mean_ret20']}%")      # -3% 보다 커야함
        # print(f"  최근 30일 중 양봉 비율              ( > 30%): {row['pos30_ratio']}%")
        # print(f"  3개월 종가 최저 대비 최고 등락률 (30% ~ 80%): {row['three_m_chg_rate']}%" )    # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
        # print(f"  3개월 종가 최고 대비 오늘 등락률   ( > -40%): {row['today_chg_rate']}%")     # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
        # print(f"  3개월 주봉 첫주 대비 이번주 등락률 ( > -20%): {row['pct_vs_firstweek']}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
        # print(f"  지난주 대비 등락률: {row['pct_vs_lastweek']}%")
        print(f"  오늘 등락률        : {row['today_pct']}%")
        print(f"  검증 등락률(max)   : {row['validation_chg_rate']}%")
        # print(f"  검증 등락률1       : {row['validation_chg_rate1']}%")
        # print(f"  검증 등락률2       : {row['validation_chg_rate2']}%")
        # print(f"  검증 등락률3       : {row['validation_chg_rate3']}%")
        # print(f"  검증 등락률4       : {row['validation_chg_rate4']}%")
        # print(f"  검증 등락률5       : {row['validation_chg_rate5']}%")
        # print(f"  검증 등락률6       : {row['validation_chg_rate6']}%")
        # print(f"  검증 등락률7       : {row['validation_chg_rate7']}%")
    #     print(f"  조건             : {row['cond']}")


    print('shortfall_cnt', shortfall_cnt)
    print('up_cnt', up_cnt)
    if shortfall_cnt+up_cnt==0:
        total_up_rate=0
    else:
        total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100

        # CSV 저장
        pd.DataFrame(rows).to_csv('csv/low_result_us_7.csv', index=False) # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
        df = pd.read_csv("csv/low_result_us_7.csv")
        saved = sort_csv_by_today_desc(
            in_path=r"csv/low_result_us_7.csv",
            out_path=r"csv/low_result_us_7_desc.csv",
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
    print('\n그래프 생성 완료')

    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"7-1_find_low_point_us.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

