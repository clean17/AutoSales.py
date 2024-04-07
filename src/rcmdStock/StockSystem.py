# 필요한 라이브러리를 불러옵니다.
from PyQt5 import QtCore, QtGui, QtWidgets
from pykrx import stock
import pandas as pd
from datetime import datetime
from PyQt5 import uic

# PyQt5에서 변환한 UI 클래스를 불러옵니다.
# form_class = uic.loadUiType("StockSystem.ui")[0]
from StockSystemUI import Ui_MainWindow

# PyQt5 애플리케이션 클래스를 작성합니다.
class StockSystemApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.search_stocks)

    def search_stocks(self):
        per_threshold = float(self.lineEdit.text())
        div_threshold = float(self.lineEdit_2.text())
        today = datetime.now().strftime('%Y%m%d')
        last_year = (datetime.now() - pd.DateOffset(years=1)).strftime('%Y%m%d')
        
        # 코스피 200 종목 리스트를 가져옵니다.
        tickers = stock.get_index_portfolio_deposit_file("1028")

        # 조건에 맞는 종목을 저장할 리스트입니다.
        matched_stocks = []

        for i, ticker in enumerate(tickers, start=1):
            print(i, ticker)
            # OHLCV 데이터와 펀더멘탈 데이터를 가져옵니다.
            df = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
            fundamentals = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)
            last_fundamentals = fundamentals.iloc[-1]

            # 조건에 맞는지 확인합니다.
            if last_fundamentals['PER'] >= per_threshold and last_fundamentals['DIV'] >= div_threshold:
                # 이동평균선 계산
                ma5 = df['종가'].rolling(window=5).mean().iloc[-1]
                ma20 = df['종가'].rolling(window=20).mean().iloc[-1]
                ma60 = df['종가'].rolling(window=60).mean().iloc[-1]

                # 정배열 조건 확인
                if ma5 > ma20 > ma60:
                    # 볼린저 밴드 계산
                    ma20 = df['종가'].rolling(window=20).mean()
                    std20 = df['종가'].rolling(window=20).std()
                    upper_band = ma20 + (std20 * 2)
                    lower_band = ma20 - (std20 * 2)

                    # 볼린저 밴드 조건 확인
                    if df['종가'].iloc[-1] < upper_band.iloc[-1] and df['종가'].iloc[-1] > lower_band.iloc[-1]:
                        matched_stocks.append((ticker, stock.get_market_ticker_name(ticker)))

        # 결과를 텍스트 에디터에 출력합니다.
        result_text = "\n".join(["{}, {}".format(ticker, name) for ticker, name in matched_stocks])
        self.textEdit.setText(result_text)

# 애플리케이션을 실행합니다.
app = QtWidgets.QApplication([])
window = StockSystemApp()
window.show()
app.exec_()
