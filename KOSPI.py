import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash

'''
1: 365 5 1차 필터링
2: 180 5 2차 필터링
'''
CONDITION = 2

# Set random seed for reproducibility
# tf.random.set_seed(42)
DROPOUT = 0.3

# 시작 종목 인덱스 ( 중단된 경우 다시 시작용 )
# count = 0
# 예측 기간
PREDICTION_PERIOD = 5
# 예측 성장률 (기본값 : 5)
EXPECTED_GROWTH_RATE = 5

# 데이터 수집 기간
if CONDITION == 1:
    DATA_COLLECTION_PERIOD = 365
elif CONDITION == 2:
    DATA_COLLECTION_PERIOD = 180
else:
    DATA_COLLECTION_PERIOD = 180

# EarlyStopping
EARLYSTOPPING_PATIENCE = 10 #숏 10, 롱 20
# 데이터셋 크기 ( 타겟 3일: 20, 5-7일: 30~50, 10일: 40~60, 15일: 50~90)
LOOK_BACK = 30
# 반복 횟수 ( 5일: 100, 7일: 150, 10일: 200, 15일: 300)
EPOCHS_SIZE = 100
BATCH_SIZE = 32

AVERAGE_VOLUME = 20000
AVERAGE_TRADING_VALUE = 1400000000

# 그래프 저장 경로
output_dir = 'D:\\kospi_stocks'
# 모델 저장 경로
# 기존 models는 LOOK_BACK = 60인 KOSPI 학습 모델이다
# model_dir = 'kospi_30_models'

if CONDITION == 1:
    model_dir = 'kospi_kosdaq_30(5)365_rmsprop_models' # 신규모델
elif CONDITION == 2:
    model_dir = 'kospi_kosdaq_30(5)180_rmsprop_models' # 신규모델
else:
    model_dir = 'kospi_kosdaq_30(5)180_rmsprop_models_128'

today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


# tickers = stock.get_market_ticker_list(market="KOSPI")

tickers_kospi = stock.get_market_ticker_list(market="KOSPI")
tickers_kosdaq = stock.get_market_ticker_list(market="KOSDAQ")


if CONDITION == 1:
    tickers = tickers_kospi + tickers_kosdaq # 전체
elif CONDITION == 2:
    tickers = None # 선택한 배열



# model_dir = os.path.join(output_dir, 'models')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터와 기본적인 재무 데이터를 가져온다
# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     if 'PER' not in fundamental.columns:
#         print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
#         fundamental['PER'] = 0  # PER 열이 없는 경우 0으로 채움
#
#     # PER 값이 NaN인 경우 0으로 채움
#     fundamental['PER'] = fundamental['PER'].fillna(0)
#     data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
#     return data

# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     daily_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker) # 기본 일별
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     # 'PER' 컬럼이 존재하는지 먼저 확인
#     if 'PER' not in daily_fundamental.columns:
#         # 일별 데이터에서 PER 정보가 없으면 월별 데이터 요청
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PER' in monthly_fundamental.columns:
#             # 월별 PER 정보를 일별 데이터에 매핑
#             daily_fundamental['PER'] = monthly_fundamental['PER'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             # 월별 PER 정보도 없는 경우 0으로 처리
#             daily_fundamental['PER'] = 0
#     else:
#         # 일별 PER 데이터 사용, NaN 값 0으로 채우기
#         daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#
#     # PER 데이터가 없으면 0으로 채우기
#     if 'PER' not in daily_fundamental.columns or daily_fundamental['PER'].isnull().all():
#         print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
#         daily_fundamental['PER'] = 0
#
#     # PER 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#     data = pd.concat([ohlcv, daily_fundamental[['PER']]], axis=1).fillna(0)
#     return data

# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     daily_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker) # 기본 일별
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     # 'PER' 컬럼이 존재하는지 먼저 확인
#     if 'PER' not in daily_fundamental.columns:
#         # 일별 데이터에서 PER 정보가 없으면 월별 데이터 요청
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PER' in monthly_fundamental.columns:
#             # 월별 PER 정보를 일별 데이터에 매핑
#             daily_fundamental['PER'] = monthly_fundamental['PER'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             daily_fundamental['PER'] = 0
#     else:
#         daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#
#     # 'PBR' 컬럼이 존재하는지 먼저 확인
#     if 'PBR' not in daily_fundamental.columns:
#         monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
#         if 'PBR' in monthly_fundamental.columns:
#             daily_fundamental['PBR'] = monthly_fundamental['PBR'].reindex(daily_fundamental.index, method='ffill')
#         else:
#             daily_fundamental['PBR'] = 0
#     else:
#         daily_fundamental['PBR'] = daily_fundamental['PBR'].fillna(0)
#
#     # PER 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
#     # PBR 값이 NaN인 경우 0으로 채움
#     daily_fundamental['PBR'] = daily_fundamental['PBR'].fillna(0)
#
#     # 필요한 데이터만 선택하여 결합
#     data = pd.concat([ohlcv[['종가', '거래량']], daily_fundamental[['PER', 'PBR']]], axis=1).fillna(0)
#
#     return data

def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    data = ohlcv[['종가', '저가', '고가', '거래량']]
    return data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        Y.append(dataset[i+look_back, 0])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = tf.keras.Sequential()

    # 30일 훈련.. 예측이 안맞는걸까 장이 안좋을걸까
    # model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    # model.add(Dropout(0.2))
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(16, activation='relu'))


    model.add((LSTM(256, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(DROPOUT))

    # 두 번째 LSTM 층
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT))

    # 세 번째 LSTM 층
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(DROPOUT))

    # Dense 레이어
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(16, activation='relu'))

    # 출력 레이어
    model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mean_squared_error') # mes

    '''
    학습률을 각 파라미터에 맞게 조정하는 방식에서, 평균 제곱을 기반으로 학습률을 조정합니다. 
    특히, 시계열 데이터와 같이 그라디언트가 빠르게 변하는 경우에 잘 작동합니다.
    
    초기 실험 단계에서는 Adam을 사용하는 것을 추천드립니다. 
    Adam은 많은 경우에서 좋은 성능을 보이며, 하이퍼파라미터 튜닝 없이도 비교적 안정적인 학습을 제공합니다. 
    그 후, 모델의 성능을 RMSprop과 비교하여 어떤 옵티마이저가 주어진 데이터셋과 모델 구조에서 더 나은 결과를 제공하는지 평가해보는 것이 좋습니다.
    '''
    model.compile(optimizer='rmsprop', loss='mse')
    return model

# @tf.function(reduce_retracing=True)
# def predict_model(model, data):
#     return model(data)


# 종목 코드와 이름 딕셔너리 생성


# 초기 설정
max_iterations = 5
# all_tickers = stock.get_market_ticker_list(market="KOSPI") + stock.get_market_ticker_list(market="KOSDAQ")
specific_tickers = ['011150', '000990', '000210', '375500', '017860', '006360', '001250', '009540', '267270', '010620', '322000', '042670', '267260', '204320', '001060', '092220', '003620', '001390', '058850', '058860', '011070', '037560', '051915', '079550', '010120', '000680', '229640', '108320', '035420', '181710', '456040', '010060', '178920', '005490', '010950', '034120', '002360', '011790', '402340', '361610', '100090', '285130', '011810', '465770', '002710', '007980', '002900', '037270', '000500', '012320', '002240', '009290', '017040', '017900', '083420', '014530', '073240', '000270', '004540', '008350', '004270', '090350', '006280', '145210', '000490', '001685', '084690', '084695', '128820', '014160', '069620', '000430', '006340', '006345', '002880', '012800', '006650', '001790', '012510', '004830', '145720', '460860', '001230', '004140', '002210', '008970', '014820', '163560', '000150', '454910', '000155', '336260', '192650', '092200', '000400', '020150', '286940', '011170', '001340', '035150', '007210', '003850', '003000', '026940', '100220', '090460', '003960', '014710', '009150', '005930', '001360', '010140', '068290', '003230', '005680', '004380', '009470', '011230', '001820', '000390', '007860', '200880', '021050', '004490', '336370', '004430', '126720', '011930', '031430', '006880', '008700', '002790', '267850', '010780', '001780', '018250', '078520', '007460', '003060', '900140', '097520', '006740', '271560', '017370', '105840', '016880', '000910', '047400', '077500', '000220', '008730', '025820', '457190', '007660', '081000', '103590', '033240', '194370', '462520', '018470', '185750', '011000', '033250', '035720', '323410', '377300', '281820', '007810', '005070', '138490', '003070', '450140', '120110', '024720', '020120', '014580', '004105', '214420', '363280', '005690', '028670', '010820', '022100', '058430', '047050', '352820', '071090', '036460', '010100', '161390', '053690', '042700', '105630', '014680', '011700', '014130', '052690', '009830', '272210', '082740', '004560', '004020', '005380', '011760', '008770', '378850', '010660', '093370', '000540', '003280', '060310', '265520', '211270', '241520', '180400', '214270', '083450', '297890', '440290', '278650', '403870', '095340', '035900', '151860', '035600', '060720', '061970', '309960', '060370', '417200', '104200', '024940', '218410', '419530', '036540', '048550', '067160', '289080', '048770', '057030', '040300', '078890', '399720', '198440', '217730', '114190', '094480', '024910', '348150', '098460', '029480', '043650', '035080', '402490', '186230', '204620', '282720', '049080', '407400', '121600', '247660', '039860', '405920', '190510', '242040', '138610', '330860', '348210', '376930', '142280', '234690', '348340', '144960', '085670', '064260', '340360', '039560', '032190', '020400', '008830', '048470', '027830', '129920', '045390', '108380', '078600', '054670', '067080', '298540', '317330', '077360', '263600', '194480', '263800', '261200', '005160', '075970', '033500', '088130', '005290', '025900', '131970', '223250', '060570', '362990', '110990', '214680', '376300', '092070', '104460', '105740', '066670', '187220', '383930', '418420', '300120', '214260', '171010', '443250', '228670', '199550', '281740', '277810', '215100', '090360', '108490', '328130', '058470', '277070', '042500', '219420', '439090', '195500', '098120', '093520', '100590', '446540', '235980', '041920', '086900', '140410', '059210', '363260', '250060', '288980', '142760', '095500', '254490', '218150', '049950', '207760', '059090', '418470', '064550', '314930', '323990', '382900', '006910', '226340', '014470', '406820', '064480', '288330', '089970', '439580', '126340', '419540', '146320', '083650', '042370', '148780', '032850', '072950', '452430', '419120', '419050', '009620', '000250', '437730', '027580', '091580', '263810', '411080', '252990', '294630', '100660', '014620', '015750', '148150', '017510', '036630', '053450', '321370', '108860', '208370', '068760', '290690', '032680', '066910', '357780', '036830', '328380', '253840', '094840', '099440', '330730', '352090', '253450', '408900', '033170', '025320', '215600', '065350', '416180', '162300', '290560', '138070', '243840', '036710', '160980', '099320', '088280', '050890', '109670', '066790', '222080', '352480', '264660', '297090', '359090', '060590', '123860', '125210', '074430', '092040', '083930', '339950', '099190', '040910', '461030', '451220', '031310', '114840', '027360', '059120', '158430', '053800', '131370', '140670', '347860', '293780', '900100', '052790', '174900', '255440', '102120', '238120', '019990', '270660', '389500', '060540', '056190', '246250', '041510', '109610', '096630', '039440', '288620', '058610', '203400', '195990', '445090', '355690', '021080', '200470', '038870', '101360', '086520', '247540', '038110', '036810', '083500', '205100', '317770', '092870', '455900', '419080', '348370', '069410', '291230', '290650', '170920', '058970', '373170', '179290', '033160', '122640', '322310', '036220', '226400', '080580', '440320', '394280', '131030', '232140', '112290', '273640', '273060', '122870', '332570', '403490', '032820', '041190', '215360', '101170', '103840', '046940', '457550', '396470', '074600', '104830', '014190', '030530', '382840', '336060', '330350', '036090', '097800', '036200', '086390', '142210', '048430', '206650', '032620', '054930', '056080', '240600', '221800', '179900', '146060', '044960', '302430', '073490', '272290', '264850', '054210', '009730', '115610', '351330', '047560', '239340', '418620', '123570', '091120', '039030', '389470', '277410', '450520', '049070', '048530', '071200', '333430', '094820', '950140', '389020', '234920', '289220', '110020', '095700', '159580', '147830', '033100', '079370', '033320', '417500', '204270', '452160', '094970', '418550', '089790', '199820', '038010', '080220', '067000', '036930', '072020', '000440', '228760', '144510', '119850', '388050', '109820', '036890', '285800', '261780', '278280', '362320', '094360', '293490', '452300', '451760', '078340', '063080', '307930', '263700', '214370', '192250', '115500', '093320', '073010', '053080', '199430', '432470', '272110', '032500', '089010', '220260', '391710', '123410', '183300', '089890', '360350', '355150', '045970', '384470', '950160', '033290', '448710', '015710', '432720', '365270', '060280', '372320', '115180', '445680', '405100', '900250', '237880', '139670', '219130', '360070', '044490', '124560', '095610', '089030', '393210', '199800', '057680', '064760', '131290', '388870', '150900', '047310', '368770', '441270', '027710', '087010', '389140', '105760', '009520', '189690', '039980', '041910', '234100', '445180', '321260', '053610', '237820', '032580', '051380', '031980', '347740', '137400', '128660', '376180', '087600', '161580', '347770', '417180', '166090', '136480', '003380', '365590', '126700', '256840', '041460', '054040', '032300', '030520', '052600', '114810', '430690', '078350', '091440', '024740', '005860', '059270', '084990', '170030', '460930', '039610', '061250', '097870', '078590', '115160', '175140', '200670', '243070', '037440']




if tickers is None:
    tickers = specific_tickers
else:
    specific_tickers = tickers

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}
ticker_returns = {}
saved_tickers = []

for iteration in range(max_iterations):
    print("\n")
    print(f"==== Iteration {iteration + 1}/{max_iterations} ====")

    # 디렉토리 내 파일 검색 및 휴지통으로 보내기
    for file_name in os.listdir(output_dir):
        if file_name.startswith(today):
            # print(f"Sending to trash: {file_name}")
            send2trash(os.path.join(output_dir, file_name))

    # 특정 배열을 가져왔을때 / 예를 들어 60(10)으로 가져온 배열을 40(5)로 돌리는 경우
    if iteration == 0:
        tickers = specific_tickers  # 두 번째 반복은 특정 배열로 실행
    else:
        tickers = saved_tickers  # 그 이후는 이전 반복에서 저장된 종목들

    # if iteration == 0:
    #     tickers = all_tickers  # 첫 번째 반복은 모든 종목
    # else:
    #     tickers = saved_tickers  # 그 이후는 이전 반복에서 저장된 종목들

    # 결과를 저장할 배열
    saved_tickers = []


    # for ticker in tickers[count:]:
    for count, ticker in enumerate(tickers):
    # for ticker in tickers[count:count+1]:
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        print(f"Processing {count+1}/{len(tickers)} : {stock_name} {ticker}")
        # count += 1

        data = fetch_stock_data(ticker, start_date, today)

        # 마지막 행의 데이터를 가져옴
        last_row = data.iloc[-1]
        # 종가가 0.0이거나 400원 미만인지 확인
        if last_row['종가'] == 0.0 or last_row['종가'] < 500:
            print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
            continue

        # 데이터가 충분하지 않으면 건너뜀
        if data.empty or len(data) < LOOK_BACK:
            print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
            continue

        # 일일 평균 거래량
        average_volume = data['거래량'].mean() # volume
        if average_volume <= AVERAGE_VOLUME:
            print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
            continue

        # 일일 평균 거래대금
        trading_value = data['거래량'] * data['종가']
        average_trading_value = trading_value.mean()
        if average_trading_value <= AVERAGE_TRADING_VALUE:
            formatted_value = f"{average_trading_value / 100000000:.0f}억"
            print(f"                                                        평균 거래액({formatted_value})이 부족하여 작업을 건너뜁니다.")
            continue

        todayTime = datetime.today()  # `today`를 datetime 객체로 유지

        # 3달 전의 종가와 비교
        three_months_ago_date = todayTime - pd.DateOffset(months=3)
        data_before_three_months = data.loc[:three_months_ago_date]

        if len(data_before_three_months) > 0:
            closing_price_three_months_ago = data_before_three_months.iloc[-1]['종가']
            if closing_price_three_months_ago > 0 and (last_row['종가'] < closing_price_three_months_ago * 0.72): # 30~40
                print(f"                                                        최근 종가가 3달 전의 종가보다 28% 이상 하락했으므로 작업을 건너뜁니다.")
                continue

        # 1년 전의 종가와 비교
        # 데이터를 기준으로 반복해서 날짜를 줄여가며 찾음
        # data_before_one_year = pd.DataFrame()  # 초기 빈 데이터프레임
        # days_offset = 365
        #
        # while days_offset >= 360:
        #     one_year_ago_date = todayTime - pd.DateOffset(days=days_offset)
        #     data_before_one_year = data.loc[:one_year_ago_date]
        #
        #     if not data_before_one_year.empty:  # 빈 배열이 아닌 경우
        #         break  # 조건을 만족하면 반복 종료
        #     days_offset -= 1  # 다음 날짜 시도

        # 1년 전과 비교
        # if len(data_before_one_year) > 0:
        #     closing_price_one_year_ago = data_before_one_year.iloc[-1]['종가']
        #     if closing_price_one_year_ago > 0 and (last_row['종가'] < closing_price_one_year_ago * 0.5):
        #         print(f"                                                        최근 종가가 1년 전의 종가보다 50% 이상 하락했으므로 작업을 건너뜁니다.")
        #         continue

        # 두 조건을 모두 만족하는지 확인
        # should_skip = False
        #
        # if len(data_before_three_months) > 0 and len(data_before_one_year) > 0:
        #     closing_price_three_months_ago = data_before_three_months.iloc[-1]['종가']
        #     closing_price_one_year_ago = data_before_one_year.iloc[-1]['종가']
        #
        #     if (closing_price_three_months_ago > 0 and last_row['종가'] < closing_price_three_months_ago * 0.7) and \
        #             (closing_price_one_year_ago > 0 and last_row['종가'] < closing_price_one_year_ago * 0.5):
        #         should_skip = True
        #
        # if should_skip:
        #     print(f"                                                        최근 종가가 3달 전의 종가보다 30% 이상 하락하고 1년 전의 종가보다 50% 이상 하락했으므로 작업을 건너뜁니다.")
        #     continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        X, Y = create_dataset(scaled_data, LOOK_BACK)

        # Python 객체 대신 TensorFlow 텐서를 사용
        # Convert the scaled_data to a TensorFlow tensor
        # scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
        # 30일 구간의 데이터셋, (365 - 30 + 1)-> 336개의 데이터셋
        # X, Y = create_dataset(scaled_data_tensor.numpy(), LOOK_BACK)  # numpy()로 변환하여 create_dataset 사용

        if len(X) < 2 or len(Y) < 2:
            print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다.")
            continue

        # 난수 데이터셋 분할
        # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

        model_file_path = os.path.join(model_dir, f'{ticker}_model_v2.Keras')
        if os.path.exists(model_file_path):
            model = tf.keras.models.load_model(model_file_path)
        else:
            model = create_model((X_train.shape[1], X_train.shape[2]))
            # 지금은 매번 학습할 예정이다
            # model.fit(X, Y, epochs=3, batch_size=32, verbose=1, validation_split=0.1)
            # model.save(model_file_path)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLYSTOPPING_PATIENCE,  # 지정한 에포크 동안 개선 없으면 종료
            verbose=0,
            mode='min',
            restore_best_weights=True  # 최적의 가중치를 복원
        )

        # 모델 학습
        model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=0, # 충분히 모델링 되었으므로 20번만
                  validation_data=(X_val, Y_val), callbacks=[early_stopping])

        close_scaler = MinMaxScaler()
        close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

        # 예측, 입력 X만 필요하다
        predictions = model.predict(X[-PREDICTION_PERIOD:])
        predicted_prices = close_scaler.inverse_transform(predictions).flatten()

        # 텐서 입력 사용하여 예측 실행 (권고)
        # TensorFlow가 함수를 그래프 모드로 변환하여 성능을 최적화하지만,
        # 이 과정에서 입력 데이터에 따라 미묘한 차이가 발생하거나 예상치 못한 동작을 할 수 있다

        # predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))
        # predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()

        model.save(model_file_path)

        last_close = data['종가'].iloc[-1]
        future_return = (predicted_prices[-1] / last_close - 1) * 100

        # 성장률 이상만
        if future_return < EXPECTED_GROWTH_RATE:
            continue

        if ticker in ticker_returns:
            ticker_returns[ticker].append(future_return)
        else:
            ticker_returns[ticker] = [future_return]

        saved_tickers.append(ticker)

        extended_prices = np.concatenate((data['종가'].values, predicted_prices))
        extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
        last_price = data['종가'].iloc[-1]

        plt.figure(figsize=(16, 8))
        plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
        plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
        plt.title(f'{today_us}   {stock_name} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)

        # 디렉토리 내 파일 검색 및 삭제
        for file_name in os.listdir(output_dir):
            if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
                print(f"Deleting existing file: {file_name}")
                os.remove(os.path.join(output_dir, file_name))

        # timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        # file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
        final_file_name = f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ].png'
        final_file_path = os.path.join(output_dir, final_file_name)
        # print(final_file_name)
        plt.savefig(final_file_path)
        plt.close()


    # for file_name in os.listdir(output_dir):
    #     if file_name.startswith(today):
    #         print(f"{file_name}")

results = []

for ticker in saved_tickers:
    if len(ticker_returns.get(ticker, [])) == 5:
        stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
        avg_future_return = sum(ticker_returns[ticker]) / 5
        results.append((avg_future_return, stock_name)) # 튜플

# avg_future_return을 기준으로 내림차순 정렬
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, stock_name in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} ====")

print(saved_tickers)