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
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
import pytz
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


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
    get_usd_krw_rate, get_nasdaq_symbols, add_today_change_rate

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, '../pickle_us')
output_dir = 'D:\\5below20_us'




def process_one(idx, count, ticker, exchangeRate):
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    df = pd.read_pickle(filepath)
    if df.empty:
        return

    date_str = df.index[-1].strftime("%Y%m%d")
    today_us = datetime.now(pytz.timezone('America/New_York')).strftime('%Y%m%d')

    if date_str != today_us:
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
    if round(mean_prev3, 1) * exchangeRate / 100_000_000 < 5:
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


    # 최근 10일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 변경점...  10일 +- 3일로 설정해봐야 할지도
    signal = signal_any_drop(data, 10, 4.0 ,-3.0, 'today_chg_rate') # 45/71 ---
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


    ########################################################################

    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)

    m_closes = m_data['Close']
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

    ma5_chg_rate = round(ma5_chg_rate, 2)
    ma20_chg_rate = round(ma20_chg_rate, 2)
    vol20 = round(vol20, 2)
    vol30 = round(vol30, 2)
    mean_ret20 = round(mean_ret20, 2)
    mean_ret30 = round(mean_ret30, 2)
    pos20_ratio = round(pos20_ratio*100, 2)
    pos30_ratio = round(pos30_ratio*100, 2)
    mean_prev3 = round(mean_prev3, 1)
    today_tr_val = round(today_tr_val, 1)
    chg_tr_val = round(chg_tr_val, 1)
    three_m_chg_rate = round(three_m_chg_rate, 2)
    today_chg_rate = round(today_chg_rate, 2)
    pct_vs_firstweek = round(result['pct_vs_firstweek'], 2)
    pct_vs_lastweek = round(result['pct_vs_lastweek'], 2)
    pct_vs_last2week = round(result['pct_vs_last2week'], 2)
    pct_vs_last3week = round(result['pct_vs_last3week'], 2)
    today_pct = round(data.iloc[-1]['today_chg_rate'], 1)

    # ----------------------------
    # 조건 플래그 초기화
    # ----------------------------
    cond01 = False
    cond02 = False
    cond03 = False
    cond04 = False
    cond05 = False
    cond06 = False
    cond07 = False
    cond08 = False
    cond09 = False
    cond10 = False
    cond11 = False
    cond12 = False
    cond13 = False
    cond14 = False
    cond15 = False
    cond16 = False
    cond17 = False
    cond18 = False
    cond19 = False
    cond20 = False
    cond21 = False
    cond22 = False
    cond23 = False
    cond24 = False
    cond25 = False
    cond26 = False
    cond27 = False
    cond28 = False
    cond29 = False
    cond30 = False




    # (1) hi80_rule_us_A_ratio_0_844_n32
    if pct_vs_last2week < -39.0846 and today_tr_val < 9.20874e+08:
        cond101 = True

    # (2) hi80_rule_us_B_ratio_0_806_n36
    if pct_vs_last2week < -39.0846 and today_tr_val < 1.02635e+09:
        cond102 = True

    # (3) hi80_rule_us_C_ratio_0_870_n23
    if pct_vs_last2week < -39.0846 and today_tr_val < 6.41818e+08:
        cond103 = True

    # (4) hi80_us_941_n17_last2w_lt_-39_0846_and_tr_lt_447_1m
    if pct_vs_last2week < -39.0846 and today_tr_val < 447130958.5:
        cond104 = True

    # (5) hi80_us_870_n23_last2w_lt_-39_0846_and_tr_lt_641_8m
    if pct_vs_last2week < -39.0846 and today_tr_val < 641818383.82:
        cond105 = True

    # (6) hi80_us_850_n20_last2w_lt_-39_0846_and_tr_lt_523_0m
    if pct_vs_last2week < -39.0846 and today_tr_val < 522960472.58:
        cond106 = True

    # (7) hi80_us_828_n29_last2w_lt_-39_0846_and_tr_lt_869_0m
    if pct_vs_last2week < -39.0846 and today_tr_val < 869029586.805:
        cond107 = True

    # (8) hi80_us_826_n23_last2w_lt_-33_2714_and_tr_lt_447_1m
    if pct_vs_last2week < -33.2714 and today_tr_val < 447130958.5:
        cond108 = True

    # (9) hi80_us_806_n31_last2w_lt_-29_2584_and_tr_lt_447_1m
    if pct_vs_last2week < -29.2584 and today_tr_val < 447130958.5:
        cond109 = True

    # (10) hi80_us_810_n21_todaychg_lt_-96_2615_and_ma5_gt_2_18
    if today_chg_rate < -96.2615 and ma5_chg_rate > 2.18:
        cond110 = True

    # (11) hi80_us_833_n18_tr_lt_148_2m_and_ma20_lt_-1_38
    if today_tr_val < 148169623.835 and ma20_chg_rate < -1.38:
        cond111 = True

    # (12) hi80_us_824_n17_tr_lt_148_2m_and_todaychg_lt_-46_107
    if today_tr_val < 148169623.835 and today_chg_rate < -46.107:
        cond112 = True

    # (13) hi80_us_800_n20_tr_lt_148_2m_and_todaychg_lt_-42_391
    if today_tr_val < 148169623.835 and today_chg_rate < -42.391:
        cond113 = True

    # (14) hi80_us_rule_1_r1.000_n31
    if (chg_tr_val <= 31.35 and
            ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond114 = True

    # (15) hi80_us_rule_2_r1.000_n26
    if (ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 8.415):
        cond115 = True

    # (16) hi80_us_rule_3_r1.000_n26
    if (ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 > 9.315 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond116 = True

    # (17) hi80_us_rule_4_r1.000_n21
    if (ma5_chg_rate > -11.895 and
            mean_prev3 > 2665270784 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond117 = True

    # (18) hi80_us_rule_5_r1.000_n21
    if (ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 8.93):
        cond118 = True

    # (19) hi80_us_rule_6_r0.927_n41
    if (ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond119 = True

    # (20) hi80_us_rule_7_r0.864_n22
    if (mean_ret30 > -0.945 and
            pct_vs_firstweek <= -36.175 and
            pct_vs_last2week <= -10.995 and
            pct_vs_last3week <= -16.79 and
            pct_vs_lastweek > 4.07 and
            three_m_chg_rate > 84.15 and
            today_pct <= 41.45 and
            vol30 > 3.795):
        cond120 = True

    # (21) hi80_us_rule_10_r0.857_n21
    if (chg_tr_val > 48.05 and
            mean_ret20 <= -0.655 and
            mean_ret30 > 0.135 and
            three_m_chg_rate <= 84.15 and
            today_pct <= 41.45 and
            vol30 > 3.795):
        cond121 = True

    # (22) hi80_us_rule_11_r0.850_n20
    if (ma5_chg_rate > -11.895 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek <= -75.465 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond122 = True

    # (23) hi80_us_rule_12_r0.850_n20
    if (ma5_chg_rate > -11.895 and
            mean_prev3 <= 2665270784 and
            mean_ret30 <= -0.945 and
            pct_vs_firstweek > -64.24 and
            pct_vs_last2week <= -38.495 and
            vol20 <= 19.4 and
            vol30 > 3.795):
        cond123 = True

    # --------------------------------
    # 모든 조건을 한 번에 모아서 체크
    # --------------------------------
    # ✅ 마지막에 "True인 조건 이름/설명"만 뽑기
    conditions = [
        ("cond01",  "", cond01),
        ("cond02",  "", cond02),
        ("cond03",  "", cond03),
        ("cond04",  "", cond04),
        ("cond05",  "", cond05),
        ("cond06",  "", cond06),
        ("cond07",  "", cond07),
        ("cond08",  "", cond08),
        ("cond09",  "", cond09),
        ("cond10", "", cond10),
        ("cond11", "", cond11),
        ("cond12", "", cond12),
        ("cond13", "", cond13),
        ("cond14", "", cond14),
        ("cond15", "", cond15),
        ("cond16", "", cond16),
        ("cond17", "", cond17),
        ("cond18", "", cond18),
        ("cond19", "", cond19),
        ("cond20", "", cond20),
        ("cond21", "", cond21),
        ("cond22", "", cond22),
        ("cond23", "", cond23),
        ("cond24", "", cond24),
        ("cond25", "", cond25),
        ("cond26", "", cond26),
        ("cond27", "", cond27),
        ("cond28", "", cond28),
        ("cond29", "", cond29),
        ("cond30", "", cond30),
    ]


    # True가 하나도 없으면 pass
    true_conds = [(name, desc) for name, desc, ok in conditions if ok]
    if not true_conds:
        return

    # 원하는 출력 형태 1) "cond17, cond30" 처럼 이름만
    # print(", ".join(name for name, _ in true_conds))
    print(f'{ticker}: {", ".join(name for name, _ in true_conds)}')



    # ─────────────────────────────────────────────────────────────
    # 2) 시가 총액 500억 이하 패스
    # ─────────────────────────────────────────────────────────────
    try:
        res = requests.post(
            'https://chickchick.shop/stocks/info',
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
        print(f"info 요청 실패-2: (코드: {str(ticker)}) {e}")
        pass  # 오류

    try:
        res2 = requests.post(
            'https://chickchick.shop/stocks/overview',
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
        "today" : str(data.index[-1].date()),
        # "3_months_ago": str(m_data.index[0].date()),
        # "predict_str": predict_str,                      # 상승/미달
        "ma5_chg_rate": ma5_chg_rate,                    # 5일선 기울기
        "ma20_chg_rate": ma20_chg_rate,                  # 20일선 기울기
        "vol20": vol20,                                  # 20일 평균 변동성
        "vol30": vol30,                                  # 30일 평균 변동성
        "mean_ret20": mean_ret20,                        # 20일 평균 등락률
        "mean_ret30": mean_ret30,                        # 30일 평균 등락률
        "pos20_ratio": pos20_ratio,                      # 20일 평균 양봉비율
        "pos30_ratio": pos30_ratio,                      # 30일 평균 양봉비율
        # "mean_prev3": mean_prev3,                        # 직전 3일 평균 거래대금
        # "today_tr_val": today_tr_val,                    # 오늘 거래대금
        "chg_tr_val": chg_tr_val,                        # 거래대금 변동률
        "three_m_chg_rate": three_m_chg_rate,            # 3개월 종가 최저 대비 최고 등락률
        "today_chg_rate": today_chg_rate,                # 3개월 종가 최고 대비 오늘 등락률
        "pct_vs_firstweek": pct_vs_firstweek,            # 3개월 주봉 첫주 대비 이번주 등락률
        "pct_vs_lastweek": pct_vs_lastweek,              # 저번주 대비 이번주 등락률
        "pct_vs_last2week": pct_vs_last2week,            # 2주 전 대비 이번주 등락률
        "pct_vs_last3week": pct_vs_last3week,            # 3주 전 대비 이번주 등락률
        "today_pct": today_pct,                          # 오늘등락률
    }



    today_str = str(today)
    title = f"{today_str} [{ticker}] Daily Chart"
    final_file_name = f"{today} [{ticker}].webp"
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
            'https://chickchick.shop/stocks/info',
            json={"stock_name": str(ticker)},
            timeout=10
        )
        json_data = res.json()
        product_code = json_data["result"][0]["data"]["items"][0]["productCode"]
    except Exception as e:
        print(f"info 요청 실패-4: {str(ticker)} {e}")
        pass  # 오류

    try:
        res2 = requests.post(
            'https://chickchick.shop/stocks/overview',
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
            'https://chickchick.shop/stocks/interest/insert',
            json={
                "nation": "kor",
                "stock_code": str(ticker),
                "pred_price_change_3d_pct": "",
                "yesterday_close": str(yesterday_close),
                "current_price": str(today_close),
                "today_price_change_pct": str(today_price_change_pct),
                "avg5d_trading_value": str(avg5),
                "current_trading_value": str(today_val),
                "trading_value_change_pct": str(ratio),
                "graph_file": str(final_file_name),
                "market_value": str(market_value),
                "target": "low-us",
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
    print(f'{nowTime} - 🕒 running 4-1_find_low_point_us.py...')
    print(' 10일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승')

    exchangeRate = get_usd_krw_rate()
    if exchangeRate is None:
        print('#######################   exchangeRate is None   #######################')
    else:
        print(f'#######################   exchangeRate is {exchangeRate}   #######################')

    tickers = get_nasdaq_symbols()
    # tickers = ['MNKD', 'ESPR']

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
                futures.append(executor.submit(process_one, idx, count, ticker, exchangeRate))

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
        print(f"\nProcessing {count+1}/{len(rows)} : [{row['ticker']}]")
        # print(f"  3개월 전 날짜           : {row['3_months_ago']}")
        # print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        # print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  거래대금 변동률         : {row['chg_tr_val']}%")
        # print(f"  20일선 기울기                      ( > -1.7): {row['ma20_chg_rate']}")
        print(f"  최근 20일 변동성                   ( > 1.5%): {row['vol20']}%")
        print(f"  최근 20일 평균 등락률            ( >= -0.5%): {row['mean_ret20']}%")      # -3% 보다 커야함
        # print(f"  최근 30일 중 양봉 비율              ( > 30%): {row['pos30_ratio']}%")
        print(f"  3개월 종가 최저 대비 최고 등락률 (30% ~ 80%): {row['three_m_chg_rate']}%" )    # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
        print(f"  3개월 종가 최고 대비 오늘 등락률   ( > -40%): {row['today_chg_rate']}%")     # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
        print(f"  3개월 주봉 첫주 대비 이번주 등락률 ( > -20%): {row['pct_vs_firstweek']}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
        print(f"  지난주 대비 등락률: {row['pct_vs_lastweek']}%")
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
        print(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

