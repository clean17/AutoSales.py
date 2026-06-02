'''
저점을 찾는 스크립트 (저점매수 + 반등 + 모멘텀)

signal_any_drop 를 통해서
5일선이 20일선보다 아래에 있으면서
최근 -3%이 존재 + 기준일 4% 이상 상승

수정사항:
- 오늘 제외
- 7일 전부터 어제까지의 각 거래일 기준으로 검사
- insert / 차트 생성도 해당 기준일 데이터로 처리
'''

import matplotlib
matplotlib.use("Agg")  # ✅ 비인터랙티브 백엔드 (창 안 띄움)

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from notUsed import lowscan_rules_v1 as rule1

modules = [rule1]


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import (
    get_kor_ticker_dict_list,
    add_technical_features,
    plot_candles_weekly,
    plot_candles_daily,
    drop_sparse_columns,
    drop_trading_halt_rows,
    signal_any_drop,
    low_weekly_check,
    safe_read_pickle,
    round_float_features, get_stock_name
)


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")
output_dir = 'F:\\5below20'
# output_dir = 'F:\\5below20_test'

api_cache = {}


def get_target_dates(df, lookback_days=7):
    """
    오늘 제외, lookback_days일 전부터 어제까지의 거래일 index 반환.

    예:
    오늘이 2026-05-19라면
    2026-05-12 ~ 2026-05-18 범위 내에 존재하는 거래일만 반환
    """
    today = pd.Timestamp.now().normalize()
    start_date = today - pd.Timedelta(days=lookback_days)
    end_date = today - pd.Timedelta(days=1)

    target_dates = df.loc[
        (df.index.normalize() >= start_date) &
        (df.index.normalize() <= end_date)
        ].index

    return target_dates


def get_stock_info(ticker):
    if ticker in api_cache:
        return api_cache[ticker]

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
            api_cache[ticker] = None
            return None

        product_code = result[0]["data"]["items"][0]["productCode"]

    except Exception as e:
        print(f"info 요청 실패-2: (코드: {str(ticker)}) {e}")
        api_cache[ticker] = None
        return None

    try:
        res2 = requests.post(
            'https://chickchick.kr/stocks/overview',
            json={"product_code": str(product_code)},
            timeout=10
        )
        data2 = res2.json()

        market_value = data2["result"]["marketValueKrw"]

        if market_value is None:
            print(f"overview marketValueKrw is None: {product_code}")
            api_cache[ticker] = None
            return None

        result = {
            "product_code": product_code,
            "market_value": market_value
        }

        api_cache[ticker] = result
        return result

    except Exception as e:
        print(f"overview 요청 실패-2: {e} {product_code}")
        api_cache[ticker] = None
        return None


def insert_low_point_stock(row, data, market_value, save_path):
    """
    data는 반드시 기준일 기준으로 잘린 데이터여야 함.
    즉 data.iloc[-1]이 row["today"]와 같은 날짜여야 함.
    """
    ticker = row["ticker"]

    if data is None or data.empty or len(data) < 2:
        return

    _closes = data['종가'].values
    _trading_value = data['종가'] * data['거래량']

    today_tr_val = _trading_value.iloc[-1]
    today_tr_val_avg_5d = _trading_value.iloc[-6:-1].mean()
    tr_val_ratio = today_tr_val / today_tr_val_avg_5d * 100

    today_close = _closes[-1]
    yesterday_close = _closes[-2]
    today_price_change_pct = (today_close - yesterday_close) / yesterday_close * 100
    today_price_change_pct = round(today_price_change_pct, 2)

    final_file_name = save_path.split("\\")[-1]

    try:
        requests.post(
            'https://chickchick.kr/stocks/interest/insert',
            json={
                "nation": "kor",
                "stock_code": str(ticker),
                "stock_name": str(row["stock_name"]),
                "pred_price_change_3d_pct": "",
                "yesterday_close": str(yesterday_close),
                "current_price": str(today_close),
                "today_price_change_pct": str(today_price_change_pct),
                "avg5d_trading_value": str(today_tr_val_avg_5d),
                "current_trading_value": str(today_tr_val),
                "trading_value_change_pct": str(tr_val_ratio),
                "graph_file": str(final_file_name),
                "market_value": str(market_value),
                "target": "low",
            },
            timeout=10
        )
    except Exception as e:
        print(f"progress-update 요청 실패-4-1: {e}")
        pass


def prepare_stock_dataframe(df):
    """
    종목별 원본 pickle 데이터를 공통 전처리하는 함수.
    process_ticker와 차트 생성부에서 같은 흐름을 쓰기 위해 분리.
    """
    if df is None or df.empty or len(df) < 70:
        return None

    df = add_technical_features(df)

    # 결측 제거
    df, _ = drop_sparse_columns(df, threshold=0.10, check_inf=True, inplace=True)

    # 거래정지/이상치 행 제거
    df, _ = drop_trading_halt_rows(df)

    # drop 이후 3차 생성
    df = add_technical_features(df)

    if df.empty or len(df) < 70:
        return None

    # 거래대금
    df["trading_value"] = df["종가"] * df["거래량"]

    return df


def process_ticker(ticker, tickers_dict, i):
    results = []

    stock_name = get_stock_name(tickers_dict, ticker)
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')

    if not os.path.exists(filepath):
        print(f"[process_ticker] {stock_name} ({ticker}) 파일 없음")
        return results

    df = safe_read_pickle(filepath)
    df = prepare_stock_dataframe(df)

    if df is None or df.empty or len(df) < 70:
        return None

    # 오늘 제외, 7일 전부터 어제까지의 실제 거래일만 검사
    target_dates = get_target_dates(df, lookback_days=30)

    if len(target_dates) == 0:
        return results

    for idx, target_date in enumerate(target_dates, start=1):
        # 기준일까지만 잘라서 검사
        sliced_df = df.loc[df.index <= target_date].copy()

        if sliced_df.empty or len(sliced_df) < 70:
            continue

        res = process_one_with_df(sliced_df, idx, ticker, tickers_dict)

        if res is not None:
            results.append(res)

    return results


def process_one_with_df(data, idx, ticker, tickers_dict):
    stock_name = get_stock_name(tickers_dict, ticker)

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
        "today": str(data.index[-1].date()),
        "idx": idx,
    }

    row.update(rule_features)
    row = round_float_features(row)

    # 종목코드 숫자 float 처리 방지
    row["ticker"] = ticker

    return {
        "row": row,
    }


if __name__ == "__main__":
    start = time.time()
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 4_find_low_point.py...')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())

    rows = []
    selected = pd.DataFrame()

    executor = None

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
            df = pd.DataFrame(rows_sorted)

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

        if selected.empty:
            nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f'{nowTime} - 조건에 맞는 종목 없음')

            end = time.time()
            elapsed = end - start

            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"4_find_low_point.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
            sys.exit(0)

        plot_jobs = []

        for _, row in selected.iterrows():
            ticker = row["ticker"]
            stock_name = row["stock_name"]

            filepath = os.path.join(pickle_dir, f"{ticker}.pkl")
            origin = safe_read_pickle(filepath)

            if origin is None or origin.empty:
                continue

            origin = prepare_stock_dataframe(origin)

            if origin is None or origin.empty or len(origin) < 70:
                continue

            target_date = pd.to_datetime(row["today"])

            # 기준일까지만 자른 데이터
            chart_data = origin.loc[origin.index <= target_date].copy()

            if chart_data.empty or len(chart_data) < 70:
                continue

            today = row["today"].replace("-", "")
            now_hm = datetime.now().strftime("%H:%M")
            rule_cnt = len(row["matched_rules"].split(",")) if row["matched_rules"] else 0

            title = (
                f"v1_{today} ({now_hm}) {stock_name} [{ticker}] "
                f"일봉 차트 - 기준일 등락률_{row['today_pct']}% "
                f"조건수: {rule_cnt}"
            )

            final_file_name = f"v1_{today} {stock_name} [{ticker}].webp"
            save_path = os.path.join(output_dir, final_file_name)

            ticker_data = get_stock_info(ticker)

            if (
                    ticker_data is None
                    or ticker_data.get("product_code") is None
                    or ticker_data.get("market_value") is None
            ):
                continue

            market_value = ticker_data['market_value']

            # 시가총액이 500억보다 작으면 패스
            if market_value < 50_000_000_000:
                continue

            # insert도 기준일 데이터 기준으로 처리
            insert_low_point_stock(row, chart_data, market_value, save_path)

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
            ax_d_vol = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
            ax_w_price = fig.add_subplot(gs[2, 0])
            ax_w_vol = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

            plot_candles_daily(
                job["origin"],
                show_months=4,
                title=job["title"],
                ax_price=ax_d_price,
                ax_volume=ax_d_vol,
                date_tick=5,
                # today=job["today"],
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

        if len(selected) > 0:
            nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            print(f'{nowTime} - ✅ 4_find_low_point.py 그래프 생성 완료')

        end = time.time()
        elapsed = end - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"4_find_low_point.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

    except KeyboardInterrupt:
        print("\n사용자 중지 요청 감지. 작업을 종료합니다.")

        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        raise