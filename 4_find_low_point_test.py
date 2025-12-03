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
    drop_sparse_columns, drop_trading_halt_rows, signal_any_drop


def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

def weekly_check(data: pd.DataFrame):
    # 인덱스가 날짜/시간이어야 함
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        data = data.copy()
        data.index = pd.to_datetime(data.index)

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    # 주봉 리샘플 (월~금 장 기준이면 W-FRI 권장)
    weekly = data.resample('W-FRI').agg({
        col_o: 'first',
        col_h: 'max',
        col_l: 'min',
        col_c: 'last',
        col_v: 'sum'
    }).dropna(subset=[col_c])  # 종가 없는 주 제거

    # 직전 2주 추출
    prev_close = weekly.iloc[-2][col_c]
    this_close = weekly.iloc[-1][col_c]   # 마지막 주 종가
    first      = weekly.iloc[0][col_c]    # 첫번째 주 종가

    past_min   = this_close.min()  # 이번 주 제외 과거 최저

    # 20% 이상 하락? (현재가가 과거최저의 80% 이하)
    is_drop_20 = this_close <= first * 0.8
    pct_from_first = this_close / first - 1.0  # 이번 주 종가(this_close)가 첫 번째 주 종가(first) 대비 몇 % 변했는지

    '''
    prev_close = 100
    this_close = 105

    pct = (105 / 100) - 1   # 1.05 - 1 = 0.05    >> pct = 0.05 (5% 상승)
    '''
    pct = (this_close / prev_close) - 1  # 저번주 대비 이번주 증감률
    is_higher = this_close > prev_close
    # is_drop_over_3 = pct < -0.005   # -0.5% 보다 더 하락했는가
    is_drop_over_3 = pct < -0.01   # -0.5% 보다 더 하락했는가

    return {
        "ok": True,
        "this_week_close": float(this_close),
        "last_week_close": float(prev_close),
        "pct_change": float(pct),                              # 예: -0.0312 == -3.12%
        "is_higher_than_last_week": bool(is_higher),           # 이번주 주봉이 저번주 보다 더 높은지
        "is_drop_more_than_minus3pct": bool(is_drop_over_3),   # 주봉 증감률이 기준보다 하락했는지
        "drop_over_3": pct,                                    # 저번주 대비 이번주 증감률
        "pct_vs_past_first": float(pct_from_first * 100),      # -0.22 -> -22% 하락
        "is_drop_more_than_20pct": bool(is_drop_20),           # 주봉 첫번째 대비 20% 이상 하락했는지
    }



# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)

tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['348370'] # 엔캠

'''
test_list = [1,2,3,4,5,6,7,8,9]
test_split = 0
while test_split < 9:
    test_split += 1
    print(test_list[:test_split])
    # [1]
    # [1, 2]
    # ...
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]

test_split = 0
while test_split < 9:
    test_split += 1
    print(test_list[-test_split:])
    # [9]
    # [8, 9]
    # ...
    # [1, 2, 3, 4, 5, 6, 7, 8, 9]
'''

shortfall_cnt = 0
up_cnt = 0
rows=[]

# quit()
# idx = -1
origin_idx = idx = 4
# origin_idx = idx = 2
while idx <= origin_idx+30:   # -10?까지 포함해서 돌리고, 다음 증가 전에 멈춤
    # while idx <= origin_idx:   # -10?까지 포함해서 돌리고, 다음 증가 전에 멈춤
    # while idx <= 0:   # -10?까지 포함해서 돌리고, 다음 증가 전에 멈춤
    idx += 1

    for count, ticker in enumerate(tickers):
        stock_name = tickers_dict.get(ticker, 'Unknown Stock')
        # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


        filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)

        origin = df.copy()

        # idx만큼 뒤에서 자른다 (idx가 2라면 2일 전 데이터셋)
        if idx != 0:
            data = df[:-(idx)]
            remaining_data = df[len(df)-idx:]
        else:
            data = df
        # data = df
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
        origin = add_technical_features(origin)

        # 결측 제거
        cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
        data = cleaned
        o_cleaned, o_cols_to_drop = drop_sparse_columns(origin, threshold=0.10, check_inf=True, inplace=True)
        origin = o_cleaned

        # 거래정지/이상치 행 제거
        data, removed_idx = drop_trading_halt_rows(data)
        origin, o_removed_idx = drop_trading_halt_rows(origin)


        if 'MA5' not in data.columns or 'MA20' not in data.columns:
            continue

        # 마지막 일자 5일선은 20일선보다 낮아야 한다
        ma5_today = data['MA5'].iloc[-1]
        ma20_today = data['MA20'].iloc[-1]

        if ma5_today >= ma20_today:
            continue

        # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승
        # 변경점...  10일 +- 3일로 설정해봐야 할지도
        # 변경점... -2.5% +- 0.5% 설정해봐야 할지도
        # signal = signal_any_drop(data, 10, 4.0 ,-2.5)
        signal = signal_any_drop(data, 10, 3.0 ,-2.5)
        if not signal:
            continue

        # ★★★★★ 오늘 13% 이상 오르면 패스 !!!!!!!!!!!!!!!!!!!
        if data.iloc[-1]['등락률'] > 13:
            continue


        ########################################################################

        m_data = data[-80:] # 뒤에서 x개 (4개월 정도)


        m_closes = m_data['종가']
        m_max = m_closes.max()
        m_min = m_closes.min()
        m_current = m_closes[-1]

        r_data = remaining_data[:10]
        r_closes = r_data['종가']
        r_max = r_closes.max()

        m_chg_rate=(m_max-m_min)/m_min*100              # 최근 4개월 동안의 등락률
        c_chg_rate=(m_current-m_max)/m_max*100         # 최근 4개월 최고 대비 오늘 등락률 계산
        r_chg_rate = (r_max-m_current)/m_current*100    # 검증 등락률

        # ★★★★★ 최근 변동률 최소 기준: 횡보 or 심한 변동 제외
        if m_chg_rate < 25 or m_chg_rate > 75:
            continue

        # ★★★★★ 최근 4개월 최고 대비 너무 내려 앉으면 패스 (보수적으로)
        if c_chg_rate > -10 or c_chg_rate < -40:
            continue


        result = weekly_check(m_data)
        if result["ok"]:
            # print("이번주 주봉 종가가 저번주보다 더 높음 : ", result["is_higher_than_last_week"])

            # ★★★★★ 저번주 대비 이번주 증감률 -1%보다 낮으면 패스 (아직 하락 추세)
            if result["is_drop_more_than_minus3pct"]:
                continue

            # ★★★★★ 지난주 대비 주봉 종가가 15% 이상 상승하면 패스
            if result['pct_change'] * 100 > 15:
                continue

            # 직전 날까지의 마지막 3일 거래대금 평균
            today_tr_val = trading_value.iloc[-1]
            mean_prev3 = trading_value.iloc[:-1].tail(3).mean()
            chg_tr_val = (today_tr_val-mean_prev3)/mean_prev3*100

            # ★★★★★ 3거래일 평균 거래대금 5억보다 작으면 패스
            if mean_prev3.round(1) / 100_000_000 < 5:
                continue

            # ★★★★★ 4개월 첫주 대비 이번주 등락률: 너무 하락한것 제외 (목 돌아감) ++ 너무 상승하면 안올라감 (보수적으로 승률을 올리기 위해 30에서 20으로 내림)
            if result['pct_vs_past_first'] > 20 or result['pct_vs_past_first'] < -25:
                continue

            # 50/-20/-20 조건 테스트 !!!!!!!!!!!!!!!!!!!!
            # if m_chg_rate < 60 and result['pct_vs_past_first'] < -17 and c_chg_rate < -19:
            # if m_chg_rate < 60 and result['pct_vs_past_first'] < -20 and c_chg_rate < -20:
            #     continue

            # 애매한 고점 근처 패턴 컷
            if -18 <= c_chg_rate <= -10 and result['pct_change']*100 >= 5 and data.iloc[-1]['등락률'] >= 5:
                continue

            # ★★★★★ 거래대금 변동률, ++ 너무 크면 차익실현으로 하락
            if chg_tr_val < -22 or chg_tr_val > 400:
                continue

            # ★★★★★ 오늘이 이미 거의 피크 느낌인 장대양봉
            # 오늘 등락률 9이상
            # 지난주 대비 등락률 9이상
            # 첫주 대비 이번주 등락률 5% 미만)
            if result['pct_change']*100 > 7 and data.iloc[-1]['등락률'] > 7 and result['pct_vs_past_first'] < -15:
                continue


            predict = '상승'
            if r_chg_rate < 5:
                predict = '미달'
                shortfall_cnt += 1
            else:
                # if r_chg_rate > 5:
                up_cnt += 1

            print(f"\nProcessing {count+1}/{len(tickers)} : {stock_name} [{ticker}] {predict}")
            # print(f"  직전 3일 평균 거래대금: {mean_prev3.round(1) / 100_000_000:.0f}억")
            # print(f"  오늘 거래대금         : {today_tr_val.round(1) / 100_000_000:.0f}억")
            print(f"  거래대금 변동률       : {chg_tr_val:.1f}%")
            print(f'  4개월 종가 최저 대비 최고 등락률 (25% ~ 80%): {m_chg_rate.round(1)}%', )             # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
            print(f"  4개월 종가 최고 대비 오늘 등락률   ( > -40%): {c_chg_rate:.1f}%")                   # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
            print(f"  4개월 주봉 첫주 대비 이번주 등락률 ( > -25%): {result['pct_vs_past_first']:.1f}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
            print(f"  지난주 대비 등락률: {result['pct_change']*100:.1f}%")
            print(f"  오늘 등락률       : {data.iloc[-1]['등락률']:.1f}%")
            print(f"  검증 등락률       : {r_chg_rate:.1f}%")

            rows.append({
                "ticker": ticker,
                "name": stock_name,
                "predict": predict,              # 상승/미달
                "r_chg_rate": r_chg_rate,        # 검증 등락률
                "m_chg_rate": float(m_chg_rate),
                "cur_chg_rate": float(c_chg_rate),
                "pct_vs_first": result['pct_vs_past_first'],
                "pct_vs_lastweek": result['pct_change']*100,
                "today_pct": data.iloc[-1]['등락률'],
                "chg_tr_val": chg_tr_val,
            })

            pd.DataFrame(rows).to_csv('low_result.csv')


            today_close = closes[-1]
            yesterday_close = closes[-2]
            change_pct_today = (today_close - yesterday_close) / yesterday_close * 100
            change_pct_today = round(change_pct_today, 2)
            avg5 = trading_value.iloc[-6:-1].mean()
            today_val = trading_value.iloc[-1]
            ratio = today_val / avg5 * 100
            ratio = round(ratio, 2)
            today_volatility_rate = round(data.iloc[-1]['등락률'], 2)
            drop_over_3 = result['drop_over_3']


        ########################################################################

        # 그래프 생성
        fig = plt.figure(figsize=(14, 16), dpi=150)
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

        ax_d_price = fig.add_subplot(gs[0, 0])
        ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
        ax_w_price = fig.add_subplot(gs[2, 0])
        ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

        plot_candles_daily(origin, show_months=6, title=f'{today} {stock_name} [{ticker}] {today_volatility_rate}% Daily Chart - {predict} {r_chg_rate:.1f}%',
                           ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

        plot_candles_weekly(origin, show_months=12, title="Weekly Chart",
                            ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        output_dir = 'D:\\5below20_test'
        os.makedirs(output_dir, exist_ok=True)

        final_file_name = f'{today} {stock_name} [{ticker}] {today_volatility_rate}%_{predict}.png'
        final_file_path = os.path.join(output_dir, final_file_name)
        plt.savefig(final_file_path)
        plt.close()


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


print('shortfall_cnt', shortfall_cnt)
print('up_cnt', up_cnt)
total_up_rate = up_cnt/(shortfall_cnt+up_cnt)*100
print(f"저점 매수 스크립트 결과 : {total_up_rate:.2f}%")


df = pd.read_csv("low_result.csv")

# 3% 기준 라벨
df['success'] = df['r_chg_rate'] >= 5

# 성공/실패 평균 비교
print(df.groupby('success')[['m_chg_rate','c_chg_rate','pct_vs_first',
                             'pct_vs_lastweek','today_pct','chg_tr_val']].mean())

# 예를 들어 c_chg_rate 분포 나눠보기
bins = [-40, -30, -25, -20, -15, -10]
df['cur_bin'] = pd.cut(df['cur_chg_rate'], bins)

print(df.pivot_table(index='cur_bin', columns='success', values='ticker', aggfunc='count'))