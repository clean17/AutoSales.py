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
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
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
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, low_weekly_check, extract_numbers_from_filenames


# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
output_dir = 'D:\\5below20_test'
# output_dir = 'D:\\5below20'

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7
render_graph = True


def process_one(idx, count, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    df = pd.read_pickle(filepath)
    
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

    closes = data['종가'].values
    trading_value = data['거래량'] * data['종가']


    # 직전 날까지의 마지막 3일 거래대금 평균
    today_tr_val = trading_value.iloc[-1]
    mean_prev3 = trading_value.iloc[:-1].tail(3).mean()
    if not np.isfinite(mean_prev3) or mean_prev3 == 0:
        chg_tr_val = 0.0
    else:
        chg_tr_val = (today_tr_val-mean_prev3)/mean_prev3*100

    # ★★★★★ 3거래일 평균 거래대금 5억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    if round(mean_prev3, 1) / 100_000_000 < 5:
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


    # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 4% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # signal = signal_any_drop(data, 12, 4.0 ,-3.0) # 40/55 ---
    # signal = signal_any_drop(data, 10, 4.0 ,-2.0) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.2) # 49/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.6) # 48/83
    # signal = signal_any_drop(data, 10, 4.0 ,-2.8) # 46/78
    signal = signal_any_drop(data, 10, 4.0 ,-3.0) # 45/71 ---
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
    last20_ret = data['등락률'].tail(20)           # 등락률이 % 단위라고 가정
    last30_ret = data['등락률'].tail(30)
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

    m_closes = m_data['종가']
    m_max = m_closes.max()
    m_min = m_closes.min()
    m_current = m_closes[-1]

    if remaining_data is not None:
        r_data = remaining_data[:10]
        r_closes = r_data['종가']
        r_max = r_closes.max()
        validation_chg_rate = (r_max-m_current)/m_current*100    # 검증 등락률
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
    today_pct = round(data.iloc[-1]['등락률'], 1)
    validation_chg_rate = round(validation_chg_rate, 1)
    predict_str = '상승'
    if validation_chg_rate < VALIDATION_TARGET_RETURN:
        predict_str = '미달'

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




    # 30일 변동성(vol30)이 매우 낮고,
    # 최근 2주 수익률이 12.36% 이상인 구간
    if vol30 <= 2.64 and pct_vs_last2week >= 12.36:
        cond01 = True


    # 최근 2주 수익률은 9.27% 이상으로 좋지만,
    # 3주 전 기준 수익률은 -1.69% 이하로 여전히 안 좋은 구간
    # -> 바닥권에서 돌아서는 턴어라운드 패턴
    if pct_vs_last2week >= 9.27 and pct_vs_last3week <= -1.69:
        cond02 = True


    # 3주 전 기준으로는 -4.06% 이하로 많이 눌려 있었고,
    # 최근 2주는 9.268% 이상 강한 기술적 반등
    if pct_vs_last2week >= 9.268 and pct_vs_last3week <= -4.06:
        cond03 = True


    # 20일 변동성이 낮고(vol20 <= 2.70),
    # 3주 전 대비 수익률이 8.89% 이상인
    # '저변동 + 최근 3주 우상향' 구간
    if vol20 <= 2.70 and pct_vs_last3week >= 8.89:
        cond04 = True


    # vol20 <= 2.70 이면서, 3주 전 대비 수익률(pct_vs_last3week)이 8.888% 이상
    # -> '더 타이트한 저변동 + 최근 3주 우상향' 패턴
    if vol20 <= 2.70 and pct_vs_last3week >= 8.888:
        cond05 = True


    # vol30 <= 2.36 이면서, 3주 전 대비 수익률이 5.634% 이상
    #  -> '초저변동 + 완만하지만 꾸준한 3주 우상향'
    if vol30 <= 2.36 and pct_vs_last3week >= 5.634:
        cond06 = True


    # vol30 <= 3.886 이면서, 첫 주 수익률이 68.298% 이상인 구간
    #  -> '30일 변동성은 적당히 낮고, 첫 주에 거의 급발진한 초강세 구간'
    if vol30 <= 3.886 and pct_vs_firstweek >= 68.298:
        cond07 = True


    # pct_vs_firstweek < 27.98 이면서 mean_ret20 < -1.07 이면서 mean_ret30 > -0.26
    if pct_vs_firstweek < 27.98 and mean_ret20 < -1.07 and mean_ret30 > -0.26:
        cond08 = True


    # pct_vs_firstweek < 49.8 이면서 mean_ret20 < -1.07 이면서 mean_ret30 > -0.26
    if pct_vs_firstweek < 49.8 and mean_ret20 < -1.07 and mean_ret30 > -0.26:
        cond09 = True


    # mean_ret30 > -0.26 이면서 pct_vs_lastweek < 4.51 이면서 mean_ret20 < -1.07
    if mean_ret30 > -0.26 and pct_vs_lastweek < 4.51 and mean_ret20 < -1.07:
        cond10 = True


    # mean_ret30 > -0.15 이면서 pct_vs_lastweek < 5.48 이면서 mean_ret20 < -1.07
    if mean_ret30 > -0.15 and pct_vs_lastweek < 5.48 and mean_ret20 < -1.07:
        cond11 = True


    # 최근 30일 동안 상승한 날 비율은 낮지만,
    # 30일 평균 수익률은 양수인 종목
    # → 많이 오르진 않았지만, 오를 때는 강하게 오르는 눌림 반등형
    if pos30_ratio < 36.67 and mean_ret30 > 0.26:
        cond12 = True


    # 최근 30일 상승일 비율이 높고,
    # 최근 3주 수익률이 크지만,
    # 30일 평균 수익률은 아직 과하지 않은 종목
    # → 최근에 추세가 막 살아난 초중반 상승 구간
    if pos30_ratio > 46.67 and pct_vs_last3week > 13.535 and mean_ret30 < 0.52:
        cond13 = True


    # 30일 기준 변동성이 있고,
    # 최근 20일 중 상승일 비율이 높으며,
    # 거래대금 변화가 큰 종목
    # → 단순 기술적 반등이 아닌 실제 수급이 붙은 종목
    if vol30 > 3.32 and pos20_ratio > 45.0 and chg_tr_val > 719.8:
        cond14 = True


    # 최근 20일 평균 수익률은 나빴지만,
    # 30일 평균은 크게 무너지지 않았고,
    # 최근 5일 급등 상태는 아닌 종목
    # → 바닥권에서 서서히 회복 중인 눌림 구간
    if mean_ret20 < -1.07 and mean_ret30 > -0.15 and ma5_chg_rate < 2.82:
        cond15 = True


    # 오늘 급락은 아니고,
    # 최근 5일 상승 탄력은 강하지만,
    # 첫 주에 과도하게 오르지 않은 종목
    # → 단기 모멘텀이 막 붙기 시작한 초기 상승 단계
    if today_chg_rate > -18.71 and ma5_chg_rate > 4.015 and pct_vs_firstweek < 8.91:
        cond16 = True


    # 최근 20일 동안 상승한 날은 많지 않지만,
    # 최근 2주 수익률은 매우 강하고,
    # 20일 이동1평균이 상승 중인 종목
    # → 조용하다가 한 번에 터지는 변동성 돌파형
    if pos20_ratio < 40.0 and pct_vs_last2week > 18.89 and ma20_chg_rate > 0.31:
        cond17 = True


    # 고거래대금 + 30일 평균수익률이 이미 높고,
    # 당일 상승률은 과열(급등) 수준까진 아니면서,
    # 거래대금 변화율/30일 변동성이 함께 커진 종목
    # → "강한 추세가 이어지는 중, 과열 없이 수급이 붙는 지속형"
    if (today_tr_val > 4151089792 and mean_ret30 > 0.265 and today_pct <= 7.05 and
            chg_tr_val > 30.9 and vol30 > 6.675):
        cont18 = True


    # 고거래대금이면서,
    # 30일 평균수익률은 상대적으로 낮지만(=아직 덜 올라온 편),
    # 3개월 누적 상승률이 44~52% 구간에 있고,
    # 당일 상승률이 강하게 터지는 종목
    # → "중기 추세는 이미 형성, 단기 모멘텀으로 재가속하는 돌파형"
    if (today_tr_val > 4151089792 and mean_ret30 <= 0.265 and three_m_chg_rate <= 51.9 and
            today_pct > 7.15 and three_m_chg_rate > 43.92):
        cond19 = True


    # 20일 변동성은 낮은 편(=조용함)인데,
    # 최근 3일 평균 거래대금이 크고,
    # 최근 3주 대비 수익률이 강한 종목
    # → "조용한 구간에서 수급이 들어오며 추세가 붙는 잠복-확장형"
    if vol20 <= 3.30 and mean_prev3 > 2.21162e9 and pct_vs_last3week > 8.78:
        cond20 = True


    # 30일 평균수익률은 플러스(=기본 추세는 있음)이고,
    # 최근 3일 평균 거래대금이 크지만,
    # 최근 3주 대비 수익률은 오히려 음수(=단기 조정 구간)
    # → "추세는 살아있고 조정 중 수급이 유지되는 눌림목 재시동형"
    if mean_ret30 > 0.10 and mean_prev3 > 3.22394e9 and pct_vs_last3week <= -4.458:
        cond21 = True


    # 5일 변화율이 강하게 플러스(=단기 모멘텀)이고,
    # 30일 변동성은 낮거나 제한적이며,
    # 최근 3일 평균 거래대금이 큰 종목
    # → "단기 모멘텀 + 과열 아닌 변동성 + 수급 동반의 안정 돌파형"
    if ma5_chg_rate > 2.10 and vol30 <= 3.06 and mean_prev3 > 2.21162e9:
        cond22 = True


    # 거래대금 변화율은 과도하지 않은 범위인데,
    # 당일 변화율은 크게 음수(=급락/쇼크성 하락)이고,
    # 당일 등락률은 오히려 높은 편(=위아래로 크게 흔들리는 날)
    # → "급격한 흔들림 이후 반등/변동성 이벤트가 나오는 급변동 이벤트형"
    if chg_tr_val <= 211.44 and today_chg_rate <= -34.016 and today_pct > 9.70:
        cond23 = True

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
    print(f'{stock_name} ({validation_chg_rate}): {", ".join(name for name, _ in true_conds)}')



    ########################################################################

    row = {
        "ticker": ticker,
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        # "3_months_ago": str(m_data.index[0].date()),
        "predict_str": predict_str,                      # 상승/미달
        "ma5_chg_rate": ma5_chg_rate,                    # 5일선 기울기
        "ma20_chg_rate": ma20_chg_rate,                  # 20일선 기울기
        "vol20": vol20,                                  # 20일 평균 변동성
        "vol30": vol30,                                  # 30일 평균 변동성
        "mean_ret20": mean_ret20,                        # 20일 평균 등락률
        "mean_ret30": mean_ret30,                        # 30일 평균 등락률
        "pos20_ratio": pos20_ratio,                      # 20일 평균 양봉비율
        "pos30_ratio": pos30_ratio,                      # 30일 평균 양봉비율
        "mean_prev3": mean_prev3,                        # 직전 3일 평균 거래대금
        "today_tr_val": today_tr_val,                    # 오늘 거래대금
        "chg_tr_val": chg_tr_val,                        # 거래대금 변동률
        "three_m_chg_rate": three_m_chg_rate,            # 3개월 종가 최저 대비 최고 등락률
        "today_chg_rate": today_chg_rate,                # 3개월 종가 최고 대비 오늘 등락률
        "pct_vs_firstweek": pct_vs_firstweek,            # 3개월 주봉 첫주 대비 이번주 등락률
        "pct_vs_lastweek": pct_vs_lastweek,              # 저번주 대비 이번주 등락률
        "pct_vs_last2week": pct_vs_last2week,            # 2주 전 대비 이번주 등락률
        "pct_vs_last3week": pct_vs_last3week,            # 3주 전 대비 이번주 등락률
        "today_pct": today_pct,                          # 오늘등락률
        "validation_chg_rate": validation_chg_rate,      # 검증 등락률
        "cond": {", ".join(name for name, _ in true_conds)}
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

    today_close = closes[-1]
    yesterday_close = closes[-2]
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100
    change_pct_today = round(change_pct_today, 2)
    avg5 = trading_value.iloc[-6:-1].mean()
    today_val = trading_value.iloc[-1]
    ratio = today_val / avg5 * 100
    ratio = round(ratio, 2)

    # try:
    #     res = requests.post(
    #         'https://chickchick.shop/func/stocks/info',
    #         json={"stock_name": str(ticker)},
    #         timeout=10
    #     )
    #     json_data = res.json()
    #     product_code = json_data["result"][0]["data"]["items"][0]["productCode"]
    # except Exception as e:
    #     print(f"info 요청 실패-4: {e}")
    #     pass  # 오류

    # try:
    #     res2 = requests.post(
    #         'https://chickchick.shop/func/stocks/overview',
    #         json={"product_code": str(product_code)},
    #         timeout=10
    #     )
    #     data2 = res2.json()
    #     market_value = data2["result"]["marketValueKrw"]
    #     company_code = data2["result"]["company"]["code"]
    # except Exception as e:
    #     print(f"overview 요청 실패-4(2): {e}")
    #     pass  # 오류
    #
    # try:
    #     res = requests.post(
    #         'https://chickchick.shop/func/stocks/company',
    #         json={"company_code": str(company_code)},
    #         timeout=15
    #     )
    #     json_data = res.json()
    #     category = json_data["result"]["majorList"][0]["title"]
    # except Exception as e:
    #     print(f"/func/stocks/company 요청 실패-4(3): {e}")
    #     pass  # 오류
    #
    # try:
    #     requests.post(
    #         'https://chickchick.shop/func/stocks/interest/insert',
    #         json={
    #             "nation": "kor",
    #             "stock_code": str(ticker),
    #             "stock_name": str(stock_name),
    #             "pred_price_change_3d_pct": "",
    #             "yesterday_close": str(yesterday_close),
    #             "current_price": str(today_close),
    #             "today_price_change_pct": str(change_pct_today),
    #             "avg5d_trading_value": str(avg5),
    #             "current_trading_value": str(today_val),
    #             "trading_value_change_pct": str(ratio),
    #             "graph_file": str(final_file_name),
    #             "market_value": str(market_value),
    #             "category": str(category),
    #             "target": "low",
    #         },
    #         timeout=10
    #     )
    # except Exception as e:
    #     # logging.warning(f"progress-update 요청 실패: {e}")
    #     print(f"progress-update 요청 실패-4-1: {e}")
    #     pass  # 오류


    return {
        "row": row,
        "plot_job": plot_job,
    }



if __name__ == "__main__":
    start = time.time()   # 시작 시간(초)
    nowTime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    print(f'🕒 {nowTime}: running 7_find_low_point.py...')
    print(' 10일 이상 5일선이 20일선 보다 아래에 있으면서 최근 -3%이 존재 + 오늘 4% 이상 상승')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())
    # tickers = extract_numbers_from_filenames(directory = r'D:\5below20_test\4퍼', isToday=False)

    shortfall_cnt = 0
    up_cnt = 0
    rows=[]
    plot_jobs = []

    # 10이면, 10거래일의 하루전부터, -1이면 어제
    origin_idx = idx = -1
    # origin_idx = idx = 5
    workers = os.cpu_count()
    BATCH_SIZE = 20

    # end_idx = origin_idx + 170 # 마지막 idx (05/13부터 데이터 만드는 용)
    # end_idx = origin_idx + 15 # 마지막 idx
    end_idx = origin_idx + 1 # 그날 하루만

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


    # 🔥 여기서 한 번에, 깔끔하게 출력
    for row in rows:
        print(f"\n {row['today']}   {row['stock_name']} [{row['ticker']}] {row['predict_str']}")
        # print(f"  3개월 전 날짜           : {row['3_months_ago']}")
        print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  거래대금 변동률         : {row['chg_tr_val']}%")
        # print(f"  20일선 기울기                      ( > -1.7): {row['ma20_chg_rate']}")
        print(f"  최근 20일 변동성                   ( > 1.5%): {row['vol20']}%")
        print(f"  최근 20일 평균 등락률              ( >= -3%): {row['mean_ret20']}%")      # -3% 보다 커야함
        print(f"  최근 30일 중 양봉 비율              ( > 30%): {row['pos30_ratio']}%")
        print(f"  3개월 종가 최저 대비 최고 등락률 (30% ~ 80%): {row['three_m_chg_rate']}%" )    # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
        print(f"  3개월 종가 최고 대비 오늘 등락률   ( > -40%): {row['today_chg_rate']}%")     # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
        print(f"  3개월 주봉 첫주 대비 이번주 등락률 ( > -20%): {row['pct_vs_firstweek']}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
        print(f"  지난주 대비 등락률: {row['pct_vs_lastweek']}%")
        print(f"  오늘 등락률       : {row['today_pct']}%")
        print(f"  검증 등락률       : {row['validation_chg_rate']}%")
        print(f"  조건             : {row['cond']}")


    print('shortfall_cnt', shortfall_cnt)
    print('up_cnt', up_cnt)
    if shortfall_cnt+up_cnt==0:
        total_up_rate=0
    else:
        total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100

        # CSV 저장
        # pd.DataFrame(rows).to_csv('low_result.csv')
        pd.DataFrame(rows).to_csv('low_result.csv', index=False) # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
        df = pd.read_csv("low_result.csv")

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
    print(f"총 소요 시간: {elapsed:.2f}초")
