'''
저점을 찾는 스크립트
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승
3일 평균 거래대금이 1000억 이상이면 무조건 사야한다
'''
import matplotlib
matplotlib.use("Agg")  # ✅ 비인터랙티브 백엔드 (창 안 띄움)
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import lowscan_rules_80_25_4_42 as rule1
import lowscan_rules_77_25_5_42 as rule2
modules = [rule1]
# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import _col, get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, low_weekly_check, extract_numbers_from_filenames, \
    safe_read_pickle

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, '../pickle')
output_dir = 'D:\\5below20'
# output_dir = 'D:\\5below20_test'




def process_one(idx, count, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    # df = pd.read_pickle(filepath)
    df = safe_read_pickle(filepath)

    date_str = df.index[-1].strftime("%Y%m%d")
    today = datetime.today().strftime('%Y%m%d')

    if date_str != today:
        return

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
        print('─────────────────────────────────────────────────────────────')
        print(data.index[-1].date())
        print('─────────────────────────────────────────────────────────────')


    ########################################################################

    closes = data['종가'].values
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
    ma20_today = data['MA20'].iloc[-1]
    ma20_yesterday = data['MA20'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    ma5_chg_rate = (ma5_today - ma5_yesterday) / ma5_yesterday * 100
    ma20_chg_rate = (ma20_today - ma20_yesterday) / ma20_yesterday * 100


    # 최근 10일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 변경점...  10일 +- 3일로 설정해봐야 할지도
    signal = signal_any_drop(data, 7, 3.0 ,-2.5) # 45/71 ---
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

    # ─────────────────────────────────────────────────────────────
    # 2) 시가 총액 500억 이하 패스
    # ─────────────────────────────────────────────────────────────
    try:
        res = requests.post(
            'https://chickchick.kr/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=10
        )
        json_data = res.json()
        result = json_data["result"]

        # 거래정지는 데이터를 주지 않는다
        if len(result) == 0:
            return

        product_code = result[0]["data"]["items"][0]["productCode"]

    except Exception as e:
        print(f"info 요청 실패-2: (코드: {str(ticker)}, 종목명: {stock_name}) {e}")
        pass  # 오류

    try:
        res2 = requests.post(
            'https://chickchick.kr/stocks/overview',
            json={"product_code": str(product_code)},
            timeout=10
        )
        data2 = res2.json()
        # if data2 is not None:
        market_value = data2["result"]["marketValueKrw"]
        company_code = data2["result"]["company"]["code"]

        if market_value is None:
            print(f"overview marketValueKrw is None: {product_code}")
            return

        # 시가총액이 500억보다 작으면 패스
        if (market_value < 50_000_000_000):
            return

    except Exception as e:
        print(f"overview 요청 실패-2: {e} {product_code}")
        pass  # 오류


    ########################################################################

    row = {
        "ticker": ticker,
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
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
    }


    today_str = str(today)
    title = f"{today_str} {stock_name} [{ticker}] Daily Chart"
    final_file_name = f"{today} {stock_name} [{ticker}].webp"
    os.makedirs(output_dir, exist_ok=True)
    final_file_path = os.path.join(output_dir, final_file_name)

    # 그래프 그릴 때 필요한 것만 모아서 리턴
    plot_job = {
        "origin": data,
        "today": today_str,
        "title": title,
        "save_path": final_file_path,
    }

    today_close = closes[-1]
    yesterday_close = closes[-2]
    today_price_change_pct = (today_close - yesterday_close) / yesterday_close * 100
    today_price_change_pct = round(today_price_change_pct, 2)
    avg5 = trading_value.iloc[-6:-1].mean()
    today_val = trading_value.iloc[-1]
    ratio = today_val / avg5 * 100
    ratio = round(ratio, 2)

    try:
        res = requests.post(
            'https://chickchick.kr/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=10
        )
        json_data = res.json()
        product_code = json_data["result"][0]["data"]["items"][0]["productCode"]
    except Exception as e:
        print(f"info 요청 실패-4: {str(ticker)} {stock_name} {e}")
        pass  # 오류

    try:
        res2 = requests.post(
            'https://chickchick.kr/stocks/overview',
            json={"product_code": str(product_code)},
            timeout=10
        )
        data2 = res2.json()
        market_value = data2["result"]["marketValueKrw"]
        company_code = data2["result"]["company"]["code"]
    except Exception as e:
        print(f"overview 요청 실패-4(2): {e}")
        pass  # 오류

    try:
        requests.post(
            'https://chickchick.kr/stocks/interest/insert',
            json={
                "nation": "kor",
                "stock_code": str(ticker),
                "stock_name": str(stock_name),
                "pred_price_change_3d_pct": "",
                "yesterday_close": str(yesterday_close),
                "current_price": str(today_close),
                "today_price_change_pct": str(today_price_change_pct),
                "avg5d_trading_value": str(avg5),
                "current_trading_value": str(today_val),
                "trading_value_change_pct": str(ratio),
                "graph_file": str(final_file_name),
                "market_value": str(market_value),
                "target": "low",
            },
            timeout=10
        )
    except Exception as e:
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"progress-update 요청 실패-4-1: {e}")
        pass  # 오류


    return {
        "row": row,
        "plot_job": plot_job,
    }



if __name__ == "__main__":
    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 4_find_low_point.py...')
    # print(' 10일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())

    rows=[]
    plot_jobs = []

    origin_idx = idx = -1  # 오늘 // 3 (5일 전)
    # origin_idx = idx = 1
    workers = os.cpu_count()
    # with ThreadPoolExecutor(max_workers=workers) as executor:   # GIL(Global Interpreter Lock) >> I/O가 많은 경우
    with ProcessPoolExecutor(max_workers=workers-4) as executor:   # CPU를 진짜로 병렬로 돌리고 싶으면 >> CPU연산이 많은 경우
        futures = []

        while idx <= origin_idx:
            idx += 1
            for count, ticker in enumerate(tickers):
                futures.append(executor.submit(process_one, idx, count, ticker, tickers_dict))

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
            plot_jobs.append(plot_job)


    # 🔥 여기서 한 번에, 깔끔하게 출력
    for count, row in enumerate(rows):
        print(f"\nProcessing {count+1}/{len(rows)} : {row['stock_name']} [{row['ticker']}]")
        print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  오늘 등락률       : {row['today_pct']}%")


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
                           ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

        plot_candles_weekly(job["origin"], show_months=12, title="Weekly Chart",
                            ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        plt.savefig(job["save_path"], format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    if len(plot_jobs) != 0:
        print('\n그래프 생성 완료')

    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    if elapsed > 20:
        print(f"4_find_low_point.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

