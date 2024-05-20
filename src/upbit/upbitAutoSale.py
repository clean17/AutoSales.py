import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
import pyupbit
import configparser

# 메인 윈도우 클래스
class CoinTradingSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super(CoinTradingSystem, self).__init__()
        uic.loadUi("CoinTradingSystem.ui", self)

        # Upbit 객체 초기화 (API 키 입력 부분은 UI에서 받는 것으로 가정)
        self.upbit = None

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_price)
        self.timer.start(1000)  # 1초마다 업데이트

        # 버튼 연결
        self.pushButton.clicked.connect(self.execute_order)

        # UI 엘리먼트에 접근하기
        self.lineEdit_price = self.findChild(QtWidgets.QLineEdit, 'lineEdit')
        self.lineEdit_profit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_2')
        self.lineEdit_take_profit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_3')
        self.lineEdit_stop_loss = self.findChild(QtWidgets.QLineEdit, 'lineEdit_4')

    def update_price(self):
        # 비트코인의 현재가를 업데이트
        current_price = pyupbit.get_current_price("KRW-BTC")
        formatted_price = "{:,.0f}".format(current_price)
        self.lineEdit_price.setText(formatted_price)

        # 주문 실행 후 손익 계산 로직 추가 필요

    def execute_order(self):
        # 사용자 API 키 입력
        config = configparser.ConfigParser()
        config.read('config.ini')

        access_key = config['DEFAULT']['AccessKey']
        secret_key = config['DEFAULT']['SecretKey']
        self.upbit = pyupbit.Upbit(access_key, secret_key)

        # 주문 금액 및 현재 가격
        order_amount = 10000  # 1만원
        current_price = pyupbit.get_current_price("KRW-BTC")
        btc_amount = order_amount / current_price

        # 지정가 매수 주문
        self.upbit.buy_limit_order("KRW-BTC", current_price, btc_amount)

        # 손익 계산 및 청산 기준가 비교 후 매도 주문 로직 필요
        # 매도 주문을 넣는 부분에서는 콤마를 제거하고 숫자로 변환해야 함
        take_profit_price = float(self.lineEdit_take_profit.text().replace(",", ""))
        stop_loss_price = float(self.lineEdit_stop_loss.text().replace(",", ""))
        # 조건에 따라 매도 주문 로직 추가 필요

# 애플리케이션 실행
app = QtWidgets.QApplication(sys.argv)
window = CoinTradingSystem()
window.show()
sys.exit(app.exec_())
