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
        print('─────────────────────────────────────────────────────────────')
        print(data.index[-1].date())
        print('─────────────────────────────────────────────────────────────')


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


    # 최근 12일 5일선이 20일선보다 낮은데 3% 하락이 있으면서 오늘 3% 상승 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 변경점...  10일 +- 3일로 설정해봐야 할지도
    signal = signal_any_drop(data, 10, 4.0 ,-3.0) # 45/71 ---
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

    m_chg_rate=(m_max-m_min)/m_min*100              # 최근 3개월 동안의 등락률
    c_chg_rate=(m_current-m_max)/m_max*100          # 최근 3개월 최고 대비 오늘 등락률 계산


    result = low_weekly_check(m_data)
    if result["ok"]:
        # ★★★★★ 저번주 대비 이번주 증감률 -1%보다 낮으면 패스 (아직 하락 추세) ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if result["is_drop_more_than_minus1pct"]:
            # return
            pass


    cond = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False
    cond6 = False
    cond7 = False
    cond8 = False
    cond9 = False

    # 100
    if round(mean_prev3, 1) / 100_000_000 >= 1000:
        cond5 = True

    # 60
    if mean_ret20 >= 0.2 and chg_tr_val <= 400:
        cond = True

    # 60
    if vol20 >= 4.1 and ma5_chg_rate <= 1.6:
        cond2 = True

    # 77
    if vol20 <= 2.9 and ma5_chg_rate >= 2.2:
        cond4 = True

    # 9.2는 60 // 11.2로 변경하면 80
    if vol20 <= 2.9 and round(result['pct_vs_lastweek']*100, 1) >= 9.2:
        cond3 = True

    # 70
    if vol20 <= 2.7 and round(result['pct_vs_lastweek']*100, 1) >= 10.3:
        cond7 = True

    # 60
    if vol20 <= 3.0 and round(result['pct_vs_last3week'], 1) >= 0.1:
        cond6 = True

    # 70
    if ma5_chg_rate >= 1.966 and vol30 <= 2.5:
        cond9 = True

    # 80
    if mean_ret20 <= -0.8 and pos30_ratio >= 50:
        cond8 = True

    if (cond is False and
            cond2 is False and
            cond3 is False and
            cond4 is False and
            cond5 is False and
            cond6 is False and
            cond7 is False and
            cond8 is False and
            cond9 is False):
        return



    ########################################################################

    row = {
        "ticker": ticker,
        "stock_name": stock_name,
        "today" : str(data.index[-1].date()),
        "3_months_ago": str(m_data.index[0].date()),
        "ma5_chg_rate": round(ma5_chg_rate, 2),                  # 5일선 기울기
        "ma20_chg_rate": round(ma20_chg_rate, 2),                # 20일선 기울기
        "vol20": round(vol20, 1),                                # 20일 평균 변동성
        "vol30": round(vol30, 1),                                # 30일 평균 변동성
        "mean_ret20": round(mean_ret20, 1),                      # 20일 평균 등락률
        "mean_ret30": round(mean_ret30, 1),                      # 30일 평균 등락률
        "pos20_ratio": round(pos20_ratio*100, 1),                # 20일 평균 양봉비율
        "pos30_ratio": round(pos30_ratio*100, 1),                # 30일 평균 양봉비율
        # "mean_prev3": round(mean_prev3, 1),                      # 직전 3일 평균 거래대금
        # "today_tr_val": round(today_tr_val, 1),                  # 오늘 거래대금
        "chg_tr_val": round(chg_tr_val, 1),                      # 거래대금 변동률
        "m_chg_rate": round(m_chg_rate, 1),                      # 3개월 종가 최저 대비 최고 등락률
        "c_chg_rate": round(c_chg_rate, 1),                      # 3개월 종가 최고 대비 오늘 등락률
        "pct_vs_first": round(result['pct_vs_first'], 1),   # 3개월 주봉 첫주 대비 이번주 등락률
        "pct_vs_last_oneweek": round(result['pct_vs_lastweek']*100, 1),   # 지난주 대비 등락률
        "pct_vs_lastweek": round(result['pct_vs_lastweek'], 1),            # 저번주 대비 이번주 증감률
        "pct_vs_last2week": round(result['pct_vs_last2week'], 1),          # 2주 전 대비 이번주 증감률
        "pct_vs_last3week": round(result['pct_vs_last3week'], 1),          # 3주 전 대비 이번주 증감률
        "today_pct": round(data.iloc[-1]['등락률'], 1),           # 오늘등락률
    }



    today_str = str(today)
    title = f"{today_str} {stock_name} [{ticker}] {round(data.iloc[-1]['등락률'], 2)}% Daily Chart"
    final_file_name = f"{today} {stock_name} [{ticker}] {round(data.iloc[-1]['등락률'], 2)}%.png"
    output_dir = 'D:\\5below20'
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
    change_pct_today = (today_close - yesterday_close) / yesterday_close * 100
    change_pct_today = round(change_pct_today, 2)
    avg5 = trading_value.iloc[-6:-1].mean()
    today_val = trading_value.iloc[-1]
    ratio = today_val / avg5 * 100
    ratio = round(ratio, 2)
    today_volatility_rate = round(data.iloc[-1]['등락률'], 2)
    pct_vs_lastweek = result['pct_vs_lastweek']

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

    rows=[]
    plot_jobs = []

    origin_idx = idx = -1
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
        # print(f"  3개월 전 날짜           : {row['3_months_ago']}")
        # print(f"  직전 3일 평균 거래대금  : {row['mean_prev3'] / 100_000_000:.0f}억")
        # print(f"  오늘 거래대금           : {row['today_tr_val'] / 100_000_000:.0f}억")
        print(f"  거래대금 변동률         : {row['chg_tr_val']}%")
        # print(f"  20일선 기울기                      ( > -1.7): {row['ma20_chg_rate']}")
        print(f"  최근 20일 변동성                   ( > 1.5%): {row['vol20']}%")
        print(f"  최근 20일 평균 등락률            ( >= -0.5%): {row['mean_ret20']}%")      # -3% 보다 커야함
        # print(f"  최근 30일 중 양봉 비율              ( > 30%): {row['pos30_ratio']}%")
        print(f"  3개월 종가 최저 대비 최고 등락률 (30% ~ 80%): {row['m_chg_rate']}%" )    # 30 ~ 65 선호, 28-30이하 애매, 70이상 과열
        print(f"  3개월 종가 최고 대비 오늘 등락률   ( > -40%): {row['c_chg_rate']}%")     # -10(15) ~ -25(30) 선호, -10(15)이상은 아직 고점, -25(30) 아래는 미달일 경우가 있음
        print(f"  3개월 주봉 첫주 대비 이번주 등락률 ( > -20%): {row['pct_vs_first']}%")   # -15 ~ 20 선호, -20이하는 장기 하락 추세, 30이상은 급등 끝물
        print(f"  지난주 대비 등락률: {row['pct_vs_last_oneweek']}%")
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

        plot_candles_daily(job["origin"], show_months=6, title=f'{job["title"]}',
                           ax_price=ax_d_price, ax_volume=ax_d_vol, date_tick=5)

        plot_candles_weekly(job["origin"], show_months=12, title="Weekly Chart",
                            ax_price=ax_w_price, ax_volume=ax_w_vol, date_tick=5)

        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        plt.savefig(job["save_path"])
        plt.close()
    print('\n그래프 생성 완료')

    end = time.time()     # 끝 시간(초)
    elapsed = end - start
    print(f"총 소요 시간: {elapsed:.2f}초")
