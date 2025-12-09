'''
저점을 찾는 스크립트
signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 3% 이상 상승
'''
print('signal_any_drop 를 통해서 5일선이 20일선보다 아래에 있으면서 최근 -3%이 존재 + 오늘 3% 이상 상승')

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import unicodedata
from pathlib import Path
import matplotlib.pyplot as plt
import requests

nowTime = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
print(f'        {nowTime}: running 4_find_low_point.py...')

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
    else:
        raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import get_kor_ticker_dict_list, add_technical_features, plot_candles_weekly, plot_candles_daily, \
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop, low_weekly_check


def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())


origin_idx = idx = -1
while idx <= origin_idx:
    idx += 1

    for count, ticker in enumerate(tickers):
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)

        data = df
        # print(data[-1:])


        if count == 0:
            # print(data)
            today = data.index[-1].strftime("%Y%m%d") # 마지막 인덱스
            print('\n\n\n\n\n\n─────────────────────────────────────────────────────────────')
            print(data.index[-1].date())
            print('─────────────────────────────────────────────────────────────')

        ########################################################################

        closes = data['종가'].values
        trading_value = data['거래량'] * data['종가']


        # 데이터가 부족하면 패스
        if data.empty or len(data) < 70:
            # print(f"                                                        데이터 부족 → pass")
            continue

        # 2차 생성 feature
        data = add_technical_features(data)

        # 결측 제거
        cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
        data = cleaned

        # 거래정지/이상치 행 제거
        data, removed_idx = drop_trading_halt_rows(data)


        if 'MA5' not in data.columns or 'MA20' not in data.columns:
            continue

        # 마지막 일자 5일선은 20일선보다 낮아야 한다
        ma5_today = data['MA5'].iloc[-1]
        ma20_today = data['MA20'].iloc[-1]
        ma20_yesterday = data['MA20'].iloc[-2]

        # if ma5_today >= ma20_today:
        #     continue




        # 변화율 계산 (퍼센트로 보려면 * 100)
        ma20_chg_rate = (ma20_today - ma20_yesterday) / ma20_yesterday * 100

        # ★★★★★ 20일선 기울기 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # if ma20_chg_rate < -1.7:
        #     continue


        # 직전 날까지의 마지막 3일 거래대금 평균
        today_tr_val = trading_value.iloc[-1]
        mean_prev3 = trading_value.iloc[:-1].tail(3).mean()
        if not np.isfinite(mean_prev3) or mean_prev3 == 0:
            chg_tr_val = 0.0
        else:
            chg_tr_val = (today_tr_val-mean_prev3)/mean_prev3*100

        # ★★★★★ 3거래일 평균 거래대금 5억보다 작으면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if mean_prev3.round(1) / 100_000_000 < 5:
            continue



        # ★★★★★ 거래대금 변동률, ++ 너무 크면 차익실현으로 하락 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # -25/500: 46.46%
        # -22/500: 46.22%
        # -20/500: 46.64% ★★★★★
        # -20/400: 46.05%
        # -15/500: 45.97%
        # -10/500: 46.63%
        #  -5/500: 46.60%
        #   0/500: 46.27%
        if chg_tr_val < -25 or chg_tr_val > 500:
            continue

        # ★★★★★ 오늘 15% 이상 오르면 패스 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # 10: 44.74%
        # 11: 45.41%
        # 12: 45.81%
        # 13: 46.08%
        # 14: 46.38%
        # 15: 46.63% ★★★★★
        if data.iloc[-1]['등락률'] > 15:
            continue

        # ★★★★★ 최근 20일 변동성 너무 낮으면 제외 (지루한 종목) ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        last20_ret = data['등락률'].tail(20)           # 등락률이 % 단위라고 가정
        last30_ret = data['등락률'].tail(30)
        vol20 = last20_ret.std()                      # 표준편차
        # if vol20 < 1.5:                               # 필요하면 2.0~3.0 사이로 튜닝
        #     continue

        # 4) 최근 20일 평균 등락률이 계속 마이너스인 종목 제외 (우하향 기는 느낌)
        mean_ret20 = last30_ret.mean()
        # if mean_ret20 < -0.5:                           # -3% 이하면 장기 하락 기조
        #     continue

        # 5) 최근 20일 중 양봉 비율이 30% 미만이면 제외 (계속 음봉 위주) ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        pos_ratio = (last30_ret > 0).mean()           # True 비율 => 양봉 비율
        # if pos_ratio < 0.3:
        #     continue




        # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승
        # 변경점...  10일 +- 3일로 설정해봐야 할지도
        # 변경점... -2.5% +- 0.5% 설정해봐야 할지도
        signal = signal_any_drop(data, 12, 4.0 ,-3.0)
        if not signal:
            continue

        # ★★★★★ 오늘 13% 이상 오르면 패스 !!!!!!!!!!!!!!!!!!!
        if data.iloc[-1]['등락률'] > 15:
            continue


        ########################################################################

        m_data = data[-80:] # 뒤에서 x개 (4개월 정도)


        m_closes = m_data['종가']
        m_max = m_closes.max()
        m_min = m_closes.min()
        m_current = m_closes[-1]

        m_chg_rate=(m_max-m_min)/m_min*100              # 최근 4개월 동안의 등락률
        c_chg_rate=(m_current-m_max)/m_max*100         # 최근 4개월 최고 대비 오늘 등락률 계산

        # ★★★★★ 최근 변동률 최소 기준: 횡보 or 심한 변동 제외
        if m_chg_rate < 30 or m_chg_rate > 80:
            continue

        # ★★★★★ 최근 4개월 최고 대비 너무 내려 앉으면 패스 (보수적으로)
        if c_chg_rate < -40:
            continue


        result = low_weekly_check(m_data)
        if result["ok"]:
            # print("이번주 주봉 종가가 저번주보다 더 높음 : ", result["is_higher_than_last_week"])

            # ★★★★★ 저번주 대비 이번주 증감률 -1%보다 낮으면 패스 (아직 하락 추세)
            if result["is_drop_more_than_minus1pct"]:
                continue

            # ★★★★★ 지난주 대비 주봉 종가가 15% 이상 상승하면 패스
            if result['pct_change'] * 100 > 15:
                continue




            # 50/-20/-20 조건 테스트 !!!!!!!!!!!!!!!!!!!!
            # if m_chg_rate < 60 and result['pct_vs_past_first'] < -17 and c_chg_rate < -19:
            # if m_chg_rate < 60 and result['pct_vs_past_first'] < -20 and c_chg_rate < -20:
            #     continue


            # ★★★★★ 오늘이 이미 거의 피크 느낌인 장대양봉
            # 오늘 등락률 9이상
            # 지난주 대비 등락률 9이상
            # 첫주 대비 이번주 등락률 5% 미만)
            # if result['pct_change']*100 > 7 and data.iloc[-1]['등락률'] > 7 and result['pct_vs_past_first'] < -15:
            #     continue

            if result['pct_vs_past_first'] < -25:
                continue


            print(f"\nProcessing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")
            # print(f"  직전 3일 평균 거래대금: {mean_prev3.round(1) / 100_000_000:.0f}억")
            # print(f"  오늘 거래대금         : {today_tr_val.round(1) / 100_000_000:.0f}억")
            print(f"  거래대금 변동률       : {chg_tr_val:.1f}%")
            print(f'  4개월 종가 최저 대비 최고 등락률 (25% ~ 80%): {m_chg_rate.round(1)}%', )             # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
            print(f"  4개월 종가 최고 대비 오늘 등락률   ( > -40%): {c_chg_rate:.1f}%")                   # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
            print(f"  4개월 주봉 첫주 대비 이번주 등락률 ( > -25%): {result['pct_vs_past_first']:.1f}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
            print(f"  지난주 대비 등락률: {result['pct_change']*100:.1f}%")
            print(f"  오늘 등락률       : {data.iloc[-1]['등락률']:.1f}%")


            today_close = closes[-1]
            yesterday_close = closes[-2]
            change_pct_today = (today_close - yesterday_close) / yesterday_close * 100
            change_pct_today = round(change_pct_today, 2)
            avg5 = trading_value.iloc[-6:-1].mean()
            today_val = trading_value.iloc[-1]
            ratio = today_val / avg5 * 100
            ratio = round(ratio, 2)
            today_volatility_rate = round(data.iloc[-1]['등락률'], 2)
            pct_change = result['pct_change']


        ########################################################################

        # 그래프 생성
        fig = plt.figure(figsize=(14, 16), dpi=150)
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

        ax_d_price = fig.add_subplot(gs[0, 0])
        ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
        ax_w_price = fig.add_subplot(gs[2, 0])
        ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

        plot_candles_daily(data, show_months=6, title=f'{today} {stock_name} [{ticker}] {today_volatility_rate}% Daily Chart',
                           ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

        plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                            ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        output_dir = 'D:\\5below20'
        os.makedirs(output_dir, exist_ok=True)

        final_file_name = f'{today} {stock_name} [{ticker}].png'
        final_file_path = os.path.join(output_dir, final_file_name)
        plt.savefig(final_file_path)
        plt.close()


        try:
            res = requests.post(
                'https://chickchick.shop/func/stocks/info',
                json={"stock_name": str(ticker)},
                timeout=10
            )
            json_data = res.json()
            product_code = json_data["result"][0]["data"]["items"][0]["productCode"]
        except Exception as e:
            print(f"info 요청 실패-4: {e}")
            pass  # 오류

        try:
            res2 = requests.post(
                'https://chickchick.shop/func/stocks/overview',
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
            res = requests.post(
                'https://chickchick.shop/func/stocks/company',
                json={"company_code": str(company_code)},
                timeout=15
            )
            json_data = res.json()
            category = json_data["result"]["majorList"][0]["title"]
        except Exception as e:
            print(f"/func/stocks/company 요청 실패-4(3): {e}")
            pass  # 오류

        try:
            requests.post(
                'https://chickchick.shop/func/stocks/interest',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "stock_name": str(stock_name),
                    "pred_price_change_3d_pct": "",
                    "yesterday_close": str(yesterday_close),
                    "current_price": str(today_close),
                    "today_price_change_pct": str(change_pct_today),
                    "avg5d_trading_value": str(avg5),
                    "current_trading_value": str(today_val),
                    "trading_value_change_pct": str(ratio),
                    "image_url": str(final_file_name),
                    "market_value": str(market_value),
                    "category": str(category),
                    "target": "low",
                },
                timeout=5
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"progress-update 요청 실패-4-1: {e}")
            pass  # 오류
