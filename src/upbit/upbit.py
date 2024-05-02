import pyupbit
import configparser

'''
<윈도우 환경 변수 설정>
import os

setx UPBIT_ACCESS_KEY "your_access_key"
setx UPBIT_SECRET_KEY "your_secret_key"

access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')

<리눅스>
export UPBIT_ACCESS_KEY="your_access_key"
export UPBIT_SECRET_KEY="your_secret_key"

<구성파일 + gitignore>
import configparser

 -++ config.ini
[DEFAULT]
AccessKey = your_access_key
SecretKey = your_secret_key
'''

# Upbit API 키 설정

config = configparser.ConfigParser()
config.read('config.ini')

access_key = config['DEFAULT']['AccessKey']
secret_key = config['DEFAULT']['SecretKey']
upbit = pyupbit.Upbit(access_key, secret_key)

# 매수하고자 하는 총 금액과 지정가
total_krw = 10000        # 1만 원
target_price = 80000000  # 지정가 8000만 원

# 비트코인의 현재 가격 가져오기
current_price = pyupbit.get_current_price("KRW-BTC")

# 주문 가능한 비트코인의 양 계산
btc_amount = total_krw / target_price

# 지정가 주문
result = upbit.buy_limit_order("KRW-BTC", target_price, btc_amount)

# 결과 출력
print(result)
