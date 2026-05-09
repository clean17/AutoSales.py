'''
저점을 찾는 스크립트
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승
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
import lowscan_rules_83_25_260504 as rule0
modules = [rule0]
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
    safe_read_pickle, safe_rate, to_float, round_float_features

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, '../pickle')
output_dir = 'F:\\5below20'
# output_dir = 'F:\\5below20_test'

api_cache = {}

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

    except:
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
            return

        result = {
            "product_code": product_code,
            "market_value": market_value
        }

        api_cache[ticker] = result
        return result

    except:
        print(f"overview 요청 실패-2: {e} {product_code}")
        api_cache[ticker] = None
        return None


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
        print('─────────────────────────────────────────────────────────────')
        print(data.index[-1].date())
        print('─────────────────────────────────────────────────────────────')


    ########################################################################

    closes = data['종가'].values
    trading_value = data['종가'] * data['거래량']
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 마지막 20일 거래대금 평균


    # ★★★★★ 20거래일 평균 거래대금 3억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
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
    trend_signal_pct        = _ma5_slope_3d - _ma20_slope_3d * 0.5
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
    dist_from_low_20d        = safe_rate(last['종가'], low_20d)

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
    
    trend_signal_pct
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
    
    dist_from_low_20d
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

        "trend_signal_pct": trend_signal_pct,

        "MACD_acc": MACD_acc,
        "MACD_hist_3d": MACD_hist_3d,

        "dist_from_low_20d": dist_from_low_20d,

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

    # result = low_weekly_check(m_data)

    ########################################################################

    # data에 컬럼이 없거나 NaN이면 넣기 (기존 컬럼 있으면 덮어쓸지 말지는 옵션)
    data = data.copy()
    for k, v in rule_features.items():
        data[k] = v


    ########################################################################

    row = {
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
    }
    row.update(rule_features)
    row = round_float_features(row)
    row["ticker"] = ticker  # 종목코드 숫자 float 처리 되어서 밖으로 뺌


    today_str = str(today)
    title = f"{today_str} {stock_name} [{ticker}] Daily Chart"
    final_file_name = f"{today} {stock_name} [{ticker}].webp"
    os.makedirs(output_dir, exist_ok=True)
    final_file_path = os.path.join(output_dir, final_file_name)

    # 그래프 그릴 때 필요한 것만 모아서 리턴
    plot_job = {
        "ticker": ticker,
        "origin": data,
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
    print(f'{nowTime} - 🔥🕒 running 4_find_low_point.py...')
    # print(' 10일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())

    rows = []
    plot_jobs_map = {}

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
            ticker = row["ticker"]

            rows.append(row)
            plot_jobs_map[ticker] = res["plot_job"]

    if len(rows) == 0:
        print("4_find_low_point.py - 데이터 없음")
        exit()

    if len(rows) > 0:
        df = pd.DataFrame(rows)

        df["MACD_hist_3d_rank"] = (
            df.groupby("today")["MACD_hist_3d"]
            .rank(pct=True)
            .round(4)
        )

        conditions = rule0.build_conditions(df)

        mask = pd.Series(False, index=df.index)

        for name, cond in conditions.items():
            mask |= cond

        selected = df[mask]

    print(f"\n🔥 최종 선택 종목 수: {len(selected)}")

    # 🔥 여기서 한 번에, 깔끔하게 출력
    for count, (_, row) in enumerate(selected.iterrows()):
        print(f"\nProcessing {count+1}/{len(selected)} : {row['stock_name']} [{row['ticker']}]")


    for _, row in selected.iterrows():
        ticker = row["ticker"]
        stock_name = row["stock_name"]

        job = plot_jobs_map[ticker]
        data = job["origin"]

        closes = data['종가'].values
        trading_value = data['종가'] * data['거래량']
        today_close = closes[-1]
        yesterday_close = closes[-2]
        today_price_change_pct = (today_close - yesterday_close) / yesterday_close * 100
        today_price_change_pct = round(today_price_change_pct, 2)
        avg5 = trading_value.iloc[-6:-1].mean()
        today_val = trading_value.iloc[-1]
        ratio = today_val / avg5 * 100
        ratio = round(ratio, 2)
        final_file_name = job["save_path"].split("\\")[-1]
        product_code = None

        ticker_data = get_stock_info(ticker)
        if (
            ticker_data is None
            or ticker_data.get("product_code") is None
            or ticker_data.get("market_value") is None
        ):
            continue

        market_value = ticker_data['market_value']

        # ─────────────────────────────────────────────────────────────
        # 2) 시가 총액 500억 이하 패스
        # ─────────────────────────────────────────────────────────────
        # 시가총액이 500억보다 작으면 패스
        if (market_value < 50_000_000_000):
            continue

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

        fig = plt.figure(figsize=(14, 16), dpi=150)
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

        ax_d_price = fig.add_subplot(gs[0, 0])
        ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
        ax_w_price = fig.add_subplot(gs[2, 0])
        ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

        plot_candles_daily(data, show_months=4,
                           title=job["title"],
                           ax_price=ax_d_price,
                           ax_volume=ax_d_vol,
                           date_tick=5)

        plot_candles_weekly(data, show_months=12,
                            title="Weekly Chart",
                            ax_price=ax_w_price,
                            ax_volume=ax_w_vol,
                            date_tick=5)

        plt.tight_layout()
        plt.savefig(job["save_path"], format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    if len(selected) != 0:
        print('\n그래프 생성 완료')

    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    if elapsed > 20:
        print(f"4_find_low_point.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")

