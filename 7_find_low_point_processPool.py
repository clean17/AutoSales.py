'''
저점을 찾는 스크립트
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 3% 이상 상승
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

# 목표 검증 수익률
VALIDATION_TARGET_RETURN = 7



def process_one(idx, count, ticker, tickers_dict):
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')

    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if not os.path.exists(filepath):
        print(f"[idx={idx}] {ticker} 파일 없음")
        return

    df = pd.read_pickle(filepath)

    # idx만큼 뒤에서 자른다 (idx가 2라면 2일 전 데이터셋)
    if idx != 0:
        data = df[:-idx]
        remaining_data = df[len(df)-idx:]
    else:
        data = df
        remaining_data = None

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

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 70:
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

    r_data = remaining_data[:10]
    r_closes = r_data['종가']
    r_max = r_closes.max()

    three_m_chg_rate=(m_max-m_min)/m_min*100        # 최근 3개월 동안의 등락률
    today_chg_rate=(m_current-m_max)/m_max*100      # 최근 3개월 최고 대비 오늘 등락률 계산
    validation_chg_rate = (r_max-m_current)/m_current*100    # 검증 등락률


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
    cond1 = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False
    cond6 = False
    cond7 = False
    cond8 = False
    cond9 = False
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
    cond31 = False
    cond32 = False
    cond33 = False
    cond34 = False
    cond35 = False
    cond36 = False
    cond37 = False
    cond38 = False
    cond39 = False

    # --------------------------------
    # [100] cond1 : 기본 유동성 필터
    # --------------------------------
    # 최근 3거래일 평균 거래대금이 1,000억 이상인 종목
    # -> 너무 작은 종목은 제외하고, 어느 정도 유동성이 담보된 종목만 사용
    if mean_prev3 / 100_000_000 >= 1000:
        cond1 = True

    # --------------------------------
    # [100] cond2 : ratio_ge_080
    # --------------------------------
    # pct_vs_last2week <= 6.6 : 직전 2주의 과열은 제한
    # vol30 <= 4.6, vol20 <= 4.86 : 20~30일 변동성이 전체적으로 낮은 종목만
    # pos30_ratio > 38.3 : 최근 30일 상승일 비율이 너무 나쁘지 않은 종목
    # ma5_chg_rate <= -1.88 : 단기(5일)는 조정이 나온 구간
    # -> '저변동 + 구조적으로 나쁘지 않은 종목의 단기 눌림' 베스트 케이스
    if (pct_vs_last2week <= 6.6 and
            vol30 <= 4.6 and
            vol20 <= 4.86 and
            pos30_ratio > 38.3 and
            ma5_chg_rate <= -1.88):
        cond2 = True

    # --------------------------------
    # [91] cond3 : vol20_le_2_95_and_pct_vs_last2week_ge_12_36
    # --------------------------------
    # 20일 변동성(vol20)이 낮으면서,
    # 최근 2주 수익률이 12.36% 이상인 저변동 + 강한 2주 랠리 구간
    if vol20 <= 2.95 and pct_vs_last2week >= 12.36:
        cond3 = True

    # --------------------------------
    # [83] cond4 : ma5>=1.966_and_vol30<=2.5
    # --------------------------------
    # 단기(5일) 수익률이 1.966% 이상이고,
    # 30일 변동성(vol30)이 2.5 이하인
    # '중기(30일)는 매우 안정적 + 단기 모멘텀 양호' 패턴
    if ma5_chg_rate >= 1.966 and vol30 <= 2.5:
        cond4 = True

    # --------------------------------
    # [83] cond5 : vol30_le_2_64_and_pct_vs_last2week_ge_12_36
    # --------------------------------
    # 30일 변동성(vol30)이 매우 낮고,
    # 최근 2주 수익률이 12.36% 이상인 구간
    if vol30 <= 2.64 and pct_vs_last2week >= 12.36:
        cond5 = True

    # --------------------------------
    # [83] cond6 : vol30_le_2_36_and_ma5_ge_1_887
    # --------------------------------
    # 초저변동(30일 vol30 <= 2.36) + 단기(5일) 수익률 1.887% 이상
    # -> '초저변동 + 단기 상승 모멘텀' 조합
    if vol30 <= 2.36 and ma5_chg_rate >= 1.887:
        cond6 = True

    # --------------------------------
    # [82] cond7 : firstweek_ge_20_85_and_2week_le_minus_1_992
    # --------------------------------
    # 첫 주에 20.85% 이상 강하게 오르고,
    # 최근 2주는 -1.992% 이하로 쉬거나 조정
    # -> '초기에 강하게 쏜 뒤 쉬고 있는 종목' 패턴
    if pct_vs_firstweek >= 20.85 and pct_vs_last2week <= -1.992:
        cond7 = True

    # --------------------------------
    # [80] cond8 : pct_vs_last2week_ge_9_27_and_pct_vs_last3week_le_minus_1_69
    # --------------------------------
    # 최근 2주 수익률은 9.27% 이상으로 좋지만,
    # 3주 전 기준 수익률은 -1.69% 이하로 여전히 안 좋은 구간
    # -> 바닥권에서 돌아서는 턴어라운드 패턴
    if pct_vs_last2week >= 9.27 and pct_vs_last3week <= -1.69:
        cond8 = True

    # --------------------------------
    # [80] cond9 : vol30_le_2_36_and_3week_ge_5_634
    # --------------------------------
    # 초저변동(vol30 <= 2.36) + 3주 전 대비 5.634% 이상 우상향
    if vol30 <= 2.36 and pct_vs_last3week >= 5.634:
        cond9 = True

    # --------------------------------
    # [80] cond10 : firstweek_ge_11_814_and_2week_le_minus_6_157
    # --------------------------------
    # 첫 주에는 11.814% 이상 올랐고,
    # 최근 2주에는 -6.157% 이하로 과도한 눌림
    # -> '초기 랠리 후 최근 2주 과도한 조정' 구간
    if pct_vs_firstweek >= 11.814 and pct_vs_last2week <= -6.157:
        cond10 = True

    # --------------------------------
    # [79] cond11 : firstweek_ge_minus_1_92_and_2week_le_minus_6_157
    # --------------------------------
    # 첫 주 기준으로는 크게 망가지지 않았지만(>= -1.92%),
    # 최근 2주는 -6.157% 이하로 꽤 큰 조정
    if pct_vs_firstweek >= -1.92 and pct_vs_last2week <= -6.157:
        cond11 = True

    # --------------------------------
    # [79] cond12 : 2week_ge_9_268_and_3week_le_minus_4_06
    # --------------------------------
    # 3주 전 기준으로는 -4.06% 이하로 많이 눌려 있었고,
    # 최근 2주는 9.268% 이상 강한 기술적 반등
    if pct_vs_last2week >= 9.268 and pct_vs_last3week <= -4.06:
        cond12 = True

    # --------------------------------
    # [78] cond13 : vol20<=2.7_and_week>=10.3
    # --------------------------------
    # 20일 변동성을 더 강하게 제한(vol20 <= 2.7)하면서도,
    # 직전 1주 수익률이 10.3% 이상인
    # '초저변동 + 직전 1주 급등' 구간
    if vol20 <= 2.7 and pct_vs_lastweek >= 10.3:
        cond13 = True

    # --------------------------------
    # [77] cond14 : vol20<=2.9_and_week>=11.2
    # --------------------------------
    # 20일 변동성이 낮고(vol20 <= 2.9),
    # 직전 1주 수익률이 11.2% 이상인 고순도 급등 구간
    if vol20 <= 2.9 and pct_vs_lastweek >= 11.2:
        cond14 = True

    # --------------------------------
    # [77] cond15 : mean_ret20_ge_0_47_and_firstweek_le_minus_12_358
    # --------------------------------
    # 최근 20일 평균 수익률은 양호(mean_ret20 >= 0.47)한데,
    # 첫 주에 -12.358% 이상 크게 눌린 자리
    # -> 좋은 추세 종목의 일시적 급락 구간
    if mean_ret20 >= 0.47 and pct_vs_firstweek <= -12.358:
        cond15 = True

    # --------------------------------
    # [77] cond16 : mean_ret20_le_0_19_and_2week_ge_18_282
    # --------------------------------
    # 최근 20일은 밋밋하거나 살짝 약한 편(mean_ret20 <= 0.19)이지만,
    # 최근 2주에 18.282% 이상 강하게 슈팅한 모멘텀주
    if mean_ret20 <= 0.19 and pct_vs_last2week >= 18.282:
        cond16 = True

    # --------------------------------
    # [76] cond17 : vol20_le_2_70_and_pct_vs_last3week_ge_8_89
    # --------------------------------
    # 20일 변동성이 낮고(vol20 <= 2.70),
    # 3주 전 대비 수익률이 8.89% 이상인
    # '저변동 + 최근 3주 우상향' 구간
    if vol20 <= 2.70 and pct_vs_last3week >= 8.89:
        cond17 = True

    # --------------------------------
    # [76] cond18 : vol30_le_3_17_and_pct_vs_last2week_ge_12_36
    # --------------------------------
    # 30일 변동성이 낮고(vol30 <= 3.17),
    # 최근 2주 수익률이 12.36% 이상인
    # '안정적인 종목 중 2주 기준 강한 랠리 대장 구간'
    if vol30 <= 3.17 and pct_vs_last2week >= 12.36:
        cond18 = True

    # --------------------------------
    # [75] cond19 : vol20<=2.9_and_ma5>=2.2
    # --------------------------------
    # 20일 변동성(vol20)이 낮으면서,
    # 단기(5일) 수익률이 2.2% 이상인
    # '저변동 + 단기 모멘텀 강한' 구간
    if vol20 <= 2.9 and ma5_chg_rate >= 2.2:
        cond19 = True

    # --------------------------------
    # [75] cond20 : vol20<2.953_and_week>10.374_and_2week>4.425
    # --------------------------------
    # 저변동(vol20 < 2.953) + 직전 1주 > 10.374% + 직전 2주 > 4.425%
    # -> 최근 1~2주 모두 강한 상승이 이어진 모멘텀 구간
    if (vol20 < 2.953 and
            pct_vs_lastweek > 10.374 and
            pct_vs_last2week > 4.425):
        cond20 = True

    # --------------------------------
    # [75] cond21 : mean_ret20<=-0.8_and_pos30>=50
    # --------------------------------
    # 최근 20일 평균 수익률이 -0.8 이하로 많이 눌렸지만,
    # 최근 30일 상승일 비율이 50% 이상인
    # '강한 조정 + 구조적으로는 여전히 강한 종목' 리버전 조건
    if mean_ret20 <= -0.8 and pos30_ratio >= 50:
        cond21 = True

    # --------------------------------
    # [75] cond22 : ratio_ge_075 (mean_ret20<=-0.8_and_pos30>=50)
    # --------------------------------
    # 위와 같은 리버전 + 구조적 강세 조건 (백테스트 기준 0.75 수준)
    if mean_ret20 <= -0.8 and pos30_ratio >= 50:
        cond22 = True

    # --------------------------------
    # [73] cond23 : vol20<2.953_and_week>10.374
    # --------------------------------
    # 20일 변동성이 낮고(vol20 < 2.953),
    # 직전 1주 수익률이 10.374% 이상인
    # '저변동 + 직전 1주 급등' 모멘텀 구간
    if vol20 < 2.953 and pct_vs_lastweek > 10.374:
        cond23 = True

    # --------------------------------
    # [73] cond24 : vol20_le_2_70_and_ma5_chg_rate_ge_1_89
    # --------------------------------
    # 20일 변동성이 낮으면서(vol20 <= 2.70),
    # 단기(5일) 수익률이 1.89% 이상인
    # '저변동 종목 중 단기 모멘텀 살아난 케이스'
    if vol20 <= 2.70 and ma5_chg_rate >= 1.89:
        cond24 = True

    # --------------------------------
    # [70] cond25 : ratio_ge_070 (mean_ret20<=-0.7_and_pos30>=50)
    # --------------------------------
    # mean_ret20 조건을 -0.7까지 완화하는 대신,
    # pos30_ratio를 50% 이상으로 유지하는 균형형 리버전 조건
    if mean_ret20 <= -0.7 and pos30_ratio >= 50:
        cond25 = True

    # [100]
    # vol20 <= 2.70 이면서, 3주 전 대비 수익률(pct_vs_last3week)이 8.888% 이상
    # -> '더 타이트한 저변동 + 최근 3주 우상향' 패턴
    if vol20 <= 2.70 and pct_vs_last3week >= 8.888:
        cond26 = True

    # [70]
    # vol30 <= 3.174 이면서, 최근 2주 수익률이 12.358% 이상
    # -> 위 조건보다 변동성을 조금 완화해서 종목 수를 늘린 버전
    if vol30 <= 3.174 and pct_vs_last2week >= 12.358:
        cond27 = True

    # [83]
    # vol30 <= 2.64 이면서, 3주 전 대비 수익률이 8.888% 이상
    # -> '초저변동 + 최근 3주 우상향' 패턴 (2주보다 조금 긴 추세)
    if vol30 <= 2.64 and pct_vs_last3week >= 8.888:
        cond28 = True

    # [83]
    # 최근 2주 수익률이 12.358% 이상이지만,
    # 3주 전 기준 수익률(pct_vs_last3week)은 -1.694% 이하
    # -> '3주 전 기준으로는 아직 저점 인식인데, 최근 2주에 강하게 턴한' 구간
    if pct_vs_last2week >= 12.358 and pct_vs_last3week <= -1.694:
        cond29 = True

    # [100]
    # vol30 <= 2.36 이면서, 3주 전 대비 수익률이 5.634% 이상
    #  -> '초저변동 + 완만하지만 꾸준한 3주 우상향'
    if vol30 <= 2.36 and pct_vs_last3week >= 5.634:
        cond30 = True

    # [89]
    # 최근 2주 수익률이 9.268% 이상인데,
    # 3주 전 기준 수익률은 -1.694% 이하
    #  -> '3주 전 기준으로는 아직 저점권인데, 최근 2주에 강하게 턴한 구간'
    if pct_vs_last2week >= 9.268 and pct_vs_last3week <= -1.694:
        cond31 = True

    # [86]
    # vol30 <= 3.886 이면서, 첫 주 수익률이 68.298% 이상인 구간
    #  -> '30일 변동성은 적당히 낮고, 첫 주에 거의 급발진한 초강세 구간'
    if vol30 <= 3.886 and pct_vs_firstweek >= 68.298:
        cond32 = True

    # [78]
    # 첫 주 수익률이 -21.71% 이하로 크게 빠졌고,
    # 직전 1주 수익률도 -0.862% 이하로 약한 구간
    #  -> '초기에 크게 깨지고, 최근 1주도 부진한 극저점 구간의 기술적 반등'
    if pct_vs_firstweek <= -21.71 and pct_vs_lastweek <= -0.862:
        cond33 = True

    # [71]
    # vol20 >= 6.834 이면서, 직전 1주 수익률이 -0.862% 이하인 구간
    #  -> '변동성은 큰 종목들 중에서, 최근 1주 조정이 나온 고위험 반등 후보'
    if vol20 >= 6.834 and pct_vs_lastweek <= -0.862:
        cond34 = True

    # [71]
    # pos30_ratio >= 50 이면서, 직전 1주 수익률이 11.362% 이상
    #  -> '최근 30일 중 절반 이상이 양봉 + 바로 직전 1주 강하게 급등한 추세주'
    if pos30_ratio >= 50.0 and pct_vs_lastweek >= 11.362:
        cond35 = True

    # [71]
    # pos30_ratio >= 46.67 이면서, 첫 주 수익률이 -7.774% 이하
    #  -> '구조적으로는 나쁘지 않은데(pos30 높음), 첫 주에 눌린 리버전 후보'
    if pos30_ratio >= 46.67 and pct_vs_firstweek <= -7.774:
        cond36 = True

    # [71]
    # mean_ret20 >= 0 이면서, 최근 2주 수익률이 1.426% 이하인 구간
    #  -> '20일 기준 우상향이지만, 최근 2주는 숨 고르기/조정인 추세 지속 구간'
    if mean_ret20 >= 0.0 and pct_vs_last2week <= 1.426:
        cond37 = True

    # [70]
    # 첫 주 수익률이 -7.774% 이하, 직전 1주 수익률도 -0.862% 이하
    #  -> '초기부터 계속 얻어맞은 종목들 중에서 기술적 반등이 많이 나왔던 구간'
    if pct_vs_firstweek <= -7.774 and pct_vs_lastweek <= -0.862:
        cond38 = True

    # [70]
    # mean_ret20 >= 0.412 이면서, 첫 주 수익률이 0.626% 이하
    #  -> '최근 20일 평균은 꽤 좋은데, 첫 주에는 상대적으로 덜 오른 저점 추세주'
    if mean_ret20 >= 0.412 and pct_vs_firstweek <= 0.626:
        cond39 = True

    # --------------------------------
    # 모든 조건을 한 번에 모아서 체크
    # --------------------------------
    condition_flags = [
        cond1,   # [100] 유동성 필터
        cond2,   # [100] ratio_ge_080
        cond3,   # [91] vol20_le_2_95_and_pct_vs_last2week_ge_12_36
        cond4,   # [83] ma5>=1.966_and_vol30<=2.5
        cond5,   # [83] vol30_le_2_64_and_pct_vs_last2week_ge_12_36
        cond6,   # [83] vol30_le_2_36_and_ma5_ge_1_887
        cond7,   # [82] firstweek_ge_20_85_and_2week_le_minus_1_992
        cond8,   # [80] pct_vs_last2week_ge_9_27_and_pct_vs_last3week_le_minus_1_69
        cond9,   # [80] vol30_le_2_36_and_3week_ge_5_634
        cond10,  # [80] firstweek_ge_11_814_and_2week_le_minus_6_157
        cond11,  # [79] firstweek_ge_minus_1_92_and_2week_le_minus_6_157
        cond12,  # [79] 2week_ge_9_268_and_3week_le_minus_4_06
        cond13,  # [78] vol20<=2.7_and_week>=10.3
        cond14,  # [77] vol20<=2.9_and_week>=11.2
        cond15,  # [77] mean_ret20_ge_0_47_and_firstweek_le_minus_12_358
        cond16,  # [77] mean_ret20_le_0_19_and_2week_ge_18_282
        cond17,  # [76] vol20_le_2_70_and_pct_vs_last3week_ge_8_89
        cond18,  # [76] vol30_le_3_17_and_pct_vs_last2week_ge_12_36
        cond19,  # [75] vol20<=2.9_and_ma5>=2.2
        cond20,  # [75] vol20<2.953_and_week>10.374_and_2week>4.425
        cond21,  # [75] mean_ret20<=-0.8_and_pos30>=50
        cond22,  # [75] ratio_ge_075
        cond23,  # [73] vol20<2.953_and_week>10.374
        cond24,  # [73] vol20_le_2_70_and_ma5_chg_rate_ge_1_89
        cond25,  # [70] ratio_ge_070
        cond26,  # [100] vol20<=2.70 AND 3week>=8.888
        cond27,  # [70]  vol30<=3.174 AND 2week>=12.358
        cond28,  # [83]  vol30<=2.64  AND 3week>=8.888
        cond29,  # [83]  2week>=12.358 AND 3week<=-1.694
        cond30,  # [100] vol30<=2.36  AND 3week>=5.634
        cond31,  # [89]  2week>=9.268  AND 3week<=-1.694
        cond32,  # [86]  vol30<=3.886 AND firstweek>=68.298
        cond33,  # [78]  firstweek<=-21.71 AND week<=-0.862
        cond34,  # [71]  vol20>=6.834 AND week<=-0.862
        cond35,  # [71]  pos30_ratio>=50 AND week>=11.362
        cond36,  # [71]  pos30_ratio>=46.67 AND firstweek<=-7.774
        cond37,  # [71]  mean_ret20>=0 AND 2week<=1.426
        cond38,  # [70]  firstweek<=-7.774 AND week<=-0.862
        cond39,  # [70]  mean_ret20>=0.412 AND firstweek<=0.626
    ]

    # 조건들 중 하나도 만족하지 않으면 이 종목은 스킵
    # if not any(condition_flags):
    #     return




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
        "validation_chg_rate": validation_chg_rate,      # 검증 등락률
    }


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
    final_file_name = f"{today} {stock_name} [{ticker}] {round(data.iloc[-1]['등락률'], 2)}%_{predict_str}.png"
    output_dir = 'D:\\5below20_test'
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
    #
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
    #         'https://chickchick.shop/func/stocks/interest',
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
    #             "image_url": str(final_file_name),
    #             "market_value": str(market_value),
    #             "category": str(category),
    #             "target": "low",
    #         },
    #         timeout=5
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
    print('signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 3% 이상 상승')
    nowTime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    print(f'        {nowTime}: running 4_find_low_point.py...')

    tickers_dict = get_kor_ticker_dict_list()
    tickers = list(tickers_dict.keys())
    # tickers = extract_numbers_from_filenames(directory = r'D:\5below20_test\4퍼', isToday=False)

    shortfall_cnt = 0
    up_cnt = 0
    rows=[]
    plot_jobs = []

    origin_idx = idx = 70
    workers = os.cpu_count()
    # with ThreadPoolExecutor(max_workers=workers) as executor:   # GIL(Global Interpreter Lock) >> I/O가 많은 경우
    with ProcessPoolExecutor(max_workers=workers-2) as executor:   # CPU를 진짜로 병렬로 돌리고 싶으면 >> CPU연산이 많은 경우
        futures = []

        while idx <= origin_idx + 30:
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

            if row["predict_str"] == "미달":
                shortfall_cnt += 1
            else:
                up_cnt += 1

    # 🔥 여기서 한 번에, 깔끔하게 출력
    for row in rows:
        print(f"\n {row['today']}   {row['stock_name']} [{row['ticker']}] {row['predict_str']}")
        # print(f"  3개월 전 날짜           : {row['3_months_ago']}")
        # print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        # print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  거래대금 변동률         : {row['chg_tr_val']}%")
        # print(f"  20일선 기울기                      ( > -1.7): {row['ma20_chg_rate']}")
        print(f"  최근 20일 변동성                   ( > 1.5%): {row['vol20']}%")
        print(f"  최근 20일 평균 등락률            ( >= -0.5%): {row['mean_ret20']}%")      # -3% 보다 커야함
        print(f"  최근 30일 중 양봉 비율              ( > 30%): {row['pos30_ratio']}%")
        print(f"  3개월 종가 최저 대비 최고 등락률 (30% ~ 80%): {row['three_m_chg_rate']}%" )    # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
        print(f"  3개월 종가 최고 대비 오늘 등락률   ( > -40%): {row['today_chg_rate']}%")     # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
        print(f"  3개월 주봉 첫주 대비 이번주 등락률 ( > -20%): {row['pct_vs_firstweek']}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
        print(f"  지난주 대비 등락률: {row['pct_vs_lastweek']}%")
        print(f"  오늘 등락률       : {row['today_pct']}%")
        print(f"  검증 등락률       : {row['validation_chg_rate']}%")


    print('shortfall_cnt', shortfall_cnt)
    print('up_cnt', up_cnt)
    total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100
    print(f"저점 매수 스크립트 결과 : {total_up_rate:.2f}%")


    # CSV 저장
    # pd.DataFrame(rows).to_csv('low_result.csv')
    pd.DataFrame(rows).to_csv('low_result.csv', index=False) # 인덱스 칼럼 'Unnamed: 0' 생성하지 않음
    df = pd.read_csv("low_result.csv")



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
        plt.savefig(job["save_path"])
        plt.close()
    print('그래프 생성 완료')



    end = time.time()     # 끝 시간(초)
    elapsed = end - start
    print(f"총 소요 시간: {elapsed:.2f}초")
