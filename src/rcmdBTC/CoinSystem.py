from ctypes import sizeof
from PyQt5 import QtWidgets, uic
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# ccxt 라이브러리를 사용하여 Upbit에서 데이터 가져오기
exchange = ccxt.upbit()

class CoinSystemApp(QtWidgets.QMainWindow):
    def __init__(self, ui_path):
        super().__init__()
        uic.loadUi(ui_path, self)
        self.pushButton.clicked.connect(self.search_coins)

    def search_coins(self):
        # 업비트 원화 마켓에 상장된 코인 목록 가져오기
        print('load_markets...')
        markets = exchange.load_markets()
        # markets = exchange.fetch_markets()
        print('select KRW-...', len(markets))
        # krw_markets = [market for market in markets if market.startswith('KRW-')]
        # krw_markets = [m for m in markets if m['quote'] == 'KRW']
        # symbols = [m['symbol'] for m in krw_markets]

        krw_markets = [symbol for symbol, market in markets.items() if market['quote'] == 'KRW']
        
        # 결과를 저장할 리스트
        matched_coins = []

        for i, symbol in enumerate(krw_markets, start=1):
            print(i, symbol)
            # 과거 1년치 OHLCV 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=exchange.parse8601((datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d %T')))
            if not ohlcv:  # 데이터가 비어 있으면 건너뛰기
                continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            if df.empty:  # 데이터프레임이 비어 있는 경우 건너뛰기
                continue
            
            # 이동평균선 계산
            df['ma7'] = df['close'].rolling(window=7).mean()
            df['ma30'] = df['close'].rolling(window=30).mean()
            df['ma90'] = df['close'].rolling(window=90).mean()

            # 볼린저 밴드 계산
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper'] = df['ma20'] + (df['std20'] * 2)
            df['lower'] = df['ma20'] - (df['std20'] * 2)

            # 조건 검사
            last_row = df.iloc[-1]
            if last_row['ma7'] > last_row['ma30'] > last_row['ma90'] and last_row['close'] < last_row['upper']:
                matched_coins.append(symbol)

        # 결과 출력
        self.textEdit.setText('\n'.join(matched_coins))

app = QtWidgets.QApplication([])
window = CoinSystemApp("CoinSystem.ui")
window.show()
app.exec_()
