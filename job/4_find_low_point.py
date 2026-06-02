'''
저점을 찾는 스크립트 (저점매수 + 반등 + 모멘텀)
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
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
# from low import lowscan_rules_260510_4tx as rule3, lowscan_rules_260510_5tx as rule4, lowscan_rules_260510_5t as rule2, \
#     lowscan_rules_260510_4t as rule1
#
# modules = [  # 전체 > 87% (67 / 77)
#     rule1,  # 단독 제외 > 87% (65 / 74)
#     rule2,  # 단독 제외 > 87% (67 / 77)  >> 오늘 등락률 포함한 depth5는 필요 없음
#     rule3,  # 단독 제외 > 87% (57 / 65)
#     rule4  # 단독 제외 > 87.01% (67 / 77) >> 필요 없음
# ]
import lowscan_good_buy_rules_v4_formatted as good_rules
import lowscan_stop_before_target_7_rules_formatted as stop_avoid_rules
import lowscan_target0_highprob_rules_cov4_c3_06_formatted as no_bounce_avoid_rules

GOOD_RULE_LABEL_MAP = {
    "rule_001_precision_high": "고확률",
    "rule_002_stable_forward": "안정형",
    "rule_003_coverage_expand": "커버리지형",
}

def pretty_good_rule_names(rule_names):
    if not rule_names:
        return ""

    parts = str(rule_names).split(",")

    labels = [
        GOOD_RULE_LABEL_MAP.get(x, x)
        for x in parts
        if x
    ]

    return "+".join(labels)


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, safe_read_pickle, safe_rate, round_float_features, \
    get_stock_name, is_korean_stock_business_day, get_market_features_for_row, get_ticker_info, low_weekly_check, \
    build_market_context_from_pickles

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")
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
            return

        result = {
            "product_code": product_code,
            "market_value": market_value
        }

        api_cache[ticker] = result
        return result

    except Exception as e:
        print(f"overview 요청 실패-2: {product_code} {e}")
        api_cache[ticker] = None
        return None


def insert_low_point_stock(row, data, market_value, save_path):
    ticker = row["ticker"]

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
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"progress-update 요청 실패-4-1: {e}")
        pass  # 오류


def process_ticker(ticker, tickers_dict, i, market_context):
    results = []

    stock_name = get_stock_name(tickers_dict, ticker)
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[process_ticker] {stock_name} ({ticker}) 파일 없음")
        return results

    df = safe_read_pickle(filepath)

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

    # today = df.index[-1].strftime("%Y%m%d") # 마지막 인덱스
    # if i == 1:
        # nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        # print('─────────────────────────────────────────────────────────────')
        # print(df.index[-1].date(), '  7일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -2.5% 이상 하락이 존재, 오늘 +3.3% 이상 상승')
        # print('─────────────────────────────────────────────────────────────')

    # 거래대금
    df["trading_value"] = df["종가"] * df["거래량"]

    res = process_one_with_df(df, 0, ticker, tickers_dict, market_context=market_context)
    if res is not None:
        results.append(res)

    return results


def process_one_with_df(data, idx, ticker, tickers_dict, market_context):
    info = get_ticker_info(ticker, tickers_dict)

    ticker = info["ticker"]
    stock_name = info["stock_name"]
    stock_market = info["stock_market"]
    sector_code = info["sector_code"]
    # stock_name = get_stock_name(tickers_dict, ticker)

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 70:
        return

    ########################################################################

    # closes = data['종가'].values
    trading_value = data['trading_value']                            # 거래대금
    today_tr_val = round(trading_value.iloc[-1], 2)                  # 마지막 거래일 거래대금
    today_tr_val_eok = today_tr_val / 100_000_000                    # 마지막 거래일 거래대금(억)
    mean_prev3 = round(trading_value.iloc[:-1].tail(3).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev5 = round(trading_value.iloc[:-1].tail(5).mean(), 2)    # 마지막 3일 거래대금 평균
    mean_prev20 = round(trading_value.iloc[:-1].tail(20).mean(), 2)  # 마지막 20일 거래대금 평균
    _mean_prev5_eok = mean_prev5 / 100_000_000
    _mean_prev20_eok = mean_prev20 / 100_000_000

    # ★★★★★ 오늘 거래대금 4억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if today_tr_val_eok < 4:
        return

    # ★★★★★ 20거래일 평균 거래대금 4억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if mean_prev20 / 100_000_000 < 4:
        return

    # 5일, 20일 이동평균선 없으면 패스
    REQUIRED_COLS = ["MA5", "MA20", "등락률"]

    for col in REQUIRED_COLS:
        if col not in data.columns:
            return

    """
    depth4 (진입 gate) >> 실패 줄이기
    - 오늘 +3% 이상
    - 최근 7일 하락 존재
    - 거래량 증가
    """
    # 오늘 제외 최근 7일 5일선이 20일선보다 계속 낮은데 -2.5% 하락이 있으면서 오늘 3.3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    signal = signal_any_drop(data, 7, 3.3, -2.5)
    # signal = signal_any_drop(data, 7, 2, -2.5)
    if not signal:
        return

    ########################################################################
    # feature 만들기
    last = data.iloc[-1]

    # 오늘의 5일션 변동율
    _ma5_today          = data['MA5'].iloc[-1]
    _ma5_yesterday      = data['MA5'].iloc[-2]
    ma5_chg_rate        = safe_rate(_ma5_today, _ma5_yesterday)

    ma5_recovery_rate_3d  = safe_rate(data["MA5"].iloc[-1], data["MA5"].iloc[-4])

    vol5                = data['등락률'].tail(5).std()
    vol15               = data['등락률'].tail(15).std()

    # 거래대금 변동률
    tr_value_ratio_3d = 0
    tr_value_ratio_5d = 0
    tr_value_ratio_20d = 0

    if mean_prev3 > 0 and np.isfinite(mean_prev3):
        tr_value_ratio_3d = today_tr_val / (mean_prev3 + 1e-9)

    if mean_prev5 > 0 and np.isfinite(mean_prev5):
        tr_value_ratio_5d = today_tr_val / (mean_prev5 + 1e-9)
        log_tr_value_ratio_5d = np.log1p(tr_value_ratio_5d)

    if mean_prev20 > 0 and np.isfinite(mean_prev20):
        tr_value_ratio_20d = today_tr_val / (mean_prev20 + 1e-9)

    # 최근 대비 오늘 거래대금 변동률 (중복)
    # _tr_value_ratio       = tr_value_ratio_3d * 0.4 + tr_value_ratio_5d * 0.6
    tr_val_rank_20d      = (trading_value.iloc[-20:] <= today_tr_val).mean()
    # tr_val_rank_60d    = (trading_value.iloc[-60:] <= today_tr_val).mean()

    # 최근 7거래일 최대 하락 (오늘 제외, 어제 포함)
    _recent_7_ret        = data['등락률'].iloc[-8:-1]
    max_drop_7d          = _recent_7_ret.min()

    # 오늘 등락률
    today_pct            = round(last['등락률'], 2)

    # 20일 최저/최고 대비 몇 % 올라왔는지
    _low_20d              = data['저가'].iloc[-20:].min()
    _low_60d              = data['저가'].iloc[-60:].min()
    _high_20d             = data["고가"].iloc[-20:].max()
    _high_60d             = data["고가"].iloc[-60:].max()
    dist_from_low_20d     = safe_rate(last['종가'], _low_20d)
    # dist_from_low_60d     = safe_rate(last['종가'], _low_60d)  # 3달 저가 대비 종가는 너무 멀다

    room_to_20d_high = (_high_20d / last['종가'] - 1) * 100
    room_to_60d_high = (_high_60d / last['종가'] - 1) * 100

    # 이평선 대비 종가의 변동률
    dist_to_ma5           = safe_rate(last['종가'], last['MA5'])
    dist_to_ma20          = safe_rate(last['종가'], last["MA20"])


    # 이평선 1일, 3일 전의 갭
    _ma5_ma20_gap_1d_ago  = safe_rate(data["MA5"].iloc[-2], data["MA20"].iloc[-2])
    _ma5_ma20_gap_3d_ago  = safe_rate(data["MA5"].iloc[-4], data["MA20"].iloc[-4])
    _ma5_ma20_gap         = safe_rate(last["MA5"], last["MA20"])
    _ma5_ma20_gap_chg_1d   = _ma5_ma20_gap - _ma5_ma20_gap_1d_ago
    _ma5_ma20_gap_chg_3d   = _ma5_ma20_gap - _ma5_ma20_gap_3d_ago  # 중복

    # 20일 저가~고가 구간에서 현재 종가가 어디쯤 있는지를 보는 값 (dist_to_ma20 상관 0.882)
    _close_pos_20d        = (last["종가"] - _low_20d) / (_high_20d - _low_20d + 1e-9)

    # -------------------------------
    # 지표
    # -------------------------------
    # 하루 변화량 : 빠름, 노이즈 많음, 3일 : 신호는 늦엇지만 안정된 필터용
    _MACD_hist_1d        = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-2]
    MACD_hist_3d         = data['MACD_hist'].iloc[-1] - data['MACD_hist'].iloc[-4]     # (AUC 0.545)
    # _ATR_pct              = last["ATR14"] / last["종가"] * 100  # 평균진폭 (변동성)

    # -------------------------------
    # 필터
    # -------------------------------
    BB_perc              = last['BB_perc']  # 중복
    _dist_to_high_20d     = safe_rate(last['종가'], _high_20d)
    _UltimateOsc          = last['UltimateOsc']
    _CCI14                = last['CCI14']
    _ADX14                = last['ADX14']
    _gap_pct              = (data['시가'].iloc[-1] / data['종가'].iloc[-2] - 1) * 100
    _vol_ratio_5_15      = round(vol5 / (vol15 + 1e-9), 3)   # 단기 변동성과 장기 변동성을 비교하는 비율
    _RSI_rebound          = data['RSI14'].iloc[-1] - data['RSI14'].iloc[-3]
    _close_pos_day        = (last["종가"] - last["저가"]) / (last["고가"] - last["저가"] + 1e-9)
    _rebound_power        = today_pct * (0.5 + _close_pos_day)
    _MACD_acc             = _MACD_hist_1d - (MACD_hist_3d / 3)  # MACD_hist_3d 보다 약함 (AUC 0.521)
    # MACD_hist_3d_atr_norm = MACD_hist_3d / (last["ATR14"] + 1e-9)
    _MACD_hist_3d_close_norm = MACD_hist_3d / (last["종가"] + 1e-9) * 100




    m_data = data[-60:] # 뒤에서 x개 (3개월 정도)
    m_current = m_data['종가'].iloc[-1]                               # 오늘 종가, 검증 데이터로 잘랐다면 검증 직전까지의 마지막 값 (수익률 분석 용도)

    m_closes = m_data['종가']


    result = low_weekly_check(m_data)

    lower_wick_ratio = (min(last["시가"], last["종가"]) - last["저가"]) / (last["고가"] - last["저가"] + 1e-9)

    # 윗꼬리 비율 (값이 크면 고가에서 많이 내려옴, 매도압력)
    upper_wick_ratio = (last["고가"] - max(last["시가"], last["종가"])) / (last["고가"] - last["저가"] + 1e-9)

    # ROC12_pct = last['ROC12_pct']

    # 캔들에서 몸통의 비율 (추세 강도)
    body_ratio = abs(last["종가"] - last["시가"]) / (last["고가"] - last["저가"] + 1e-9)
    # 당일 등락률(today_pct)과 거래대금(today_tr_val_eok)을 결합한 지표
    price_power_value = today_pct * np.log1p(max(today_tr_val_eok, 0))
    # 당일 캔들의 “몸통(body)” 크기와 방향성, 당일 수익률, 거래대금 규모를 결합해서 만든 일종의 매수세 강도 지표
    body_value_power = (
            max(body_ratio, 0)
            * max(today_pct, 0)
            * np.log1p(max(today_tr_val_eok, 0))
    )

    _recent_5d            = data.iloc[-5:]
    _ret5                 = _recent_5d["등락률"]
    _recent_tr_value      = _recent_5d["종가"] * _recent_5d["거래량"]

    # 이미 많이 오른 종목 제거.. 추격매수 위험 필터
    _recent_runup        = data['등락률'].iloc[-5:-1].sum()

    # 시가 대비 종가가 얼마나 회복되었는가 (AUC 0.514 - 거의 랜덤)
    intraday_return = (last["종가"] / last["시가"] - 1) * 100
    # 전일 대비 갭이 아니라 “장중 순수 매수세”.. 상광 0.929
    intraday_body_power = intraday_return * body_ratio

    _low_7d_before_today = data['저가'].iloc[-8:-1].min()
    # 현재 종가가 7일 최저가보다 얼마나 위에 있는지 비율로 표현
    rebound_from_7d_low = (last['종가'] / _low_7d_before_today - 1) * 100
    # 최근 하락폭 대비 오늘까지 얼마나 반등했는가
    rebound_vs_prior_drop = rebound_from_7d_low / (abs(max_drop_7d) + 1e-9)
    if not np.isfinite(rebound_vs_prior_drop):
        rebound_vs_prior_drop = np.nan

    today_str = str(data.index[-1].date())

    market_item = get_market_features_for_row(
        today=today_str,
        stock_market=stock_market,
        market_context=market_context,
    )

    market_today_pct = market_item["market_today_pct"]
    market_5d_pct = market_item["market_5d_pct"]
    market_breadth_up_ratio = market_item["market_breadth_up_ratio"]
    market_stock_count = market_item["market_stock_count"]

    rule_features = {
        "vol5": vol5,                                          # 성공군의 단기 변동성이 큼, 핵심 피쳐
        "vol_ratio_5_15": _vol_ratio_5_15,                     # 성공군의 최근 단기 변동성 확장이 큼, 정밀도를 올려줌, 표본 부족
        "today_pct": today_pct,                                # 성공군의 마지막 날 반등 강도가 큼
        "max_drop_7d": max_drop_7d,                            # 성공군이 더 깊게 빠졌다가 반등
        "gap_pct": _gap_pct,                                   # gap_pct는 단독 분리력은 약하지만 today_pct와 조합하면 의미가 있을 수 있음, 비단조 주의, 룰 조합에서 강함, 핵심 피쳐

        "pct_vs_lastweek": result['pct_vs_lastweek'],          # 단독 AUC는 약하지만 룰 조합에서 강함, 없으면 데드캣 상승, 주요 피쳐
        "dist_to_ma5": dist_to_ma5,                            # dist_from_low_20d, pct_vs_lastweek, dist_to_ma20와 중복이 큼, 대체 가능
        "ma5_chg_rate": ma5_chg_rate,                          # 5일선 기울기
        "BB_perc": BB_perc,                                    # 볼린저밴드 위치
        "today_tr_val_eok": today_tr_val_eok,                  # 오늘 거래대금 (억)

        "lower_wick_ratio": lower_wick_ratio,                  # 아랫꼬리 비율
        "upper_wick_ratio": upper_wick_ratio,                  # 윗꼬리 비율
        "body_ratio": body_ratio,                              # 몸통 비율
        "intraday_return": intraday_return,                    # 시가 대비 종가가 얼마나 회복되었는가
        "rebound_from_7d_low": rebound_from_7d_low,            # 현재 종가가 7일 최저가보다 얼마나 위에 있는지 비율로 표현

        "price_power_value": price_power_value,                # 당일 등락률(today_pct)과 거래대금(today_tr_val_eok)을 결합한 지표
        "body_value_power": body_value_power,                  # 당일 캔들의 “몸통(body)” 크기와 방향성, 당일 수익률, 거래대금 규모를 결합해서 만든 일종의 매수세 강도 지표
        "rebound_vs_prior_drop": rebound_vs_prior_drop,        # 최근 하락폭 대비 오늘까지 얼마나 반등했는가
        "room_to_20d_high": room_to_20d_high,                  # 주가가 어느 정도 반등했는지 또는 앞으로 상승 여력이 얼마나 있는지를 보는 피쳐
        "room_to_60d_high": room_to_60d_high,

        "market_today_pct": market_today_pct,                  # 해당 종목이 속한 시장의 당일 등락률
        "market_5d_pct": market_5d_pct,                        # 해당 종목이 속한 시장의 최근 5거래일 등락률
    }

    ########################################################################

    row = {
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "idx": idx,
    }

    row.update(rule_features)
    row = round_float_features(row)
    row["ticker"] = ticker  # 종목코드 숫자 float 처리 되어서 밖으로 뺌

    return {
        "row": row,
    }



if __name__ == "__main__":
    if not is_korean_stock_business_day(verbose=False):
        # print("한국증시 영업일이 아니므로 실행하지 않습니다.")
        sys.exit(0)

    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 4_find_low_point.py...')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())

    # print("[market_context] building...")
    market_context = build_market_context_from_pickles(
        tickers=tickers,
        tickers_dict=tickers_dict,
        pickle_dir=pickle_dir,
    )


    rows = []            # 결과 종목 데이터 저장

    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            futures = [
                executor.submit(process_ticker, ticker, tickers_dict, i, market_context)
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

        selected = pd.DataFrame()

        if len(rows) > 0:
            df = pd.DataFrame(rows)

            # =========================
            # 1) 좋은 룰: 통과해야 함
            # =========================
            good_conditions = good_rules.build_conditions(df)

            good_mask = pd.Series(False, index=df.index)
            good_rule_map = {idx: [] for idx in df.index}

            for name in good_rules.RULE_NAMES:
                if name not in good_conditions:
                    continue

                cond = pd.Series(good_conditions[name], index=df.index).fillna(False).astype(bool)

                good_mask |= cond

                for idx in df.index[cond]:
                    good_rule_map[idx].append(name)

            # =========================
            # 2) 나쁜 룰: 걸리면 제외
            # =========================
            stop_conditions = stop_avoid_rules.build_conditions(df)
            no_bounce_conditions = no_bounce_avoid_rules.build_conditions(df)

            stop_avoid_mask = pd.Series(False, index=df.index)
            no_bounce_avoid_mask = pd.Series(False, index=df.index)

            for name in stop_avoid_rules.RULE_NAMES:
                if name not in stop_conditions:
                    continue

                cond = pd.Series(stop_conditions[name], index=df.index).fillna(False).astype(bool)
                stop_avoid_mask |= cond

            for name in no_bounce_avoid_rules.RULE_NAMES:
                if name not in no_bounce_conditions:
                    continue

                cond = pd.Series(no_bounce_conditions[name], index=df.index).fillna(False).astype(bool)
                no_bounce_avoid_mask |= cond

            avoid_mask = stop_avoid_mask | no_bounce_avoid_mask

            # =========================
            # 3) 최종 선택
            # 좋은 룰 통과 AND 나쁜 룰 미통과
            # =========================
            final_mask = good_mask & (~avoid_mask)

            df["good_rule_names"] = df.index.map(
                lambda idx: ",".join(good_rule_map[idx])
            )

            df["good_rule_label"] = df["good_rule_names"].map(pretty_good_rule_names)

            df["good_rule_pass"] = good_mask.astype(int)
            df["stop_avoid_pass"] = stop_avoid_mask.astype(int)
            df["no_bounce_avoid_pass"] = no_bounce_avoid_mask.astype(int)
            df["avoid_rule_pass"] = avoid_mask.astype(int)
            df["selected"] = final_mask.astype(int)

            selected = df[final_mask].copy()


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

            # 기본값: idx == 0이면 전체 origin 사용
            chart_data = origin.copy()

            if chart_data.empty or len(chart_data) < 70:
                continue

            today = row["today"].replace("-", "")
            now_hm = datetime.now().strftime("%H:%M")
            rule_label = row.get("good_rule_label", "")

            title = (
                f"{today} ({now_hm}) {stock_name} [{ticker}] "
                f"일봉 차트 - 오늘 등락률_{row['today_pct']}% "
                f"룰: {rule_label} "
            )

            final_file_name = f"{today} {stock_name} [{ticker}].webp"
            save_path = os.path.join(output_dir, final_file_name)

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

            insert_low_point_stock(row, origin, market_value, save_path)

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
            print(f'{nowTime} - ✅ 4_find_low_poind.py 그래프 생성 완료')


        end = time.time()     # 끝 시간(초)
        elapsed = end - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        # if elapsed > 20:
        #     print(f"4_find_low_point.py 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
        # nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        # print(f'{nowTime} - ✅ 저점 종목 스캔 완료, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')


    except KeyboardInterrupt:
        print("\n사용자 중지 요청 감지. 작업을 종료합니다.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise