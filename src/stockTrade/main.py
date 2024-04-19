import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QTimer
from pykiwoom.kiwoom import Kiwoom


class StockTradingSystem(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('StockTradingSystem.ui', self)
        self.kiwoom = Kiwoom()
        self.kiwoom.CommConnect()  # Kiwoom 서버에 로그인

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_price)
        self.timer.start(1000)  # 1초마다 현재가 업데이트

        self.pushButton.clicked.connect(self.execute_order)


    def get_current_price(self, code):
        """ 종목 코드에 해당하는 현재가를 가져오는 함수 """
        # 데이터 요청
        self.kiwoom.SetInputValue("종목코드", code)
        self.kiwoom.CommRqData("opt10001_req", "opt10001", 0, "0101")

        # 이벤트 루프: 요청에 대한 응답을 기다림
        self.loop = QEventLoop()
        self.kiwoom.OnReceiveTrData.connect(self.on_receive_tr_data)
        self.loop.exec_()

    def on_receive_tr_data(self, screen_no, rqname, trcode, recordname, prev_next, data_len, error_code, message, splm_msg):
        """ 트랜잭션 데이터 수신 시 처리 """
        if rqname == "opt10001_req":
            current_price = self.kiwoom.GetCommData(trcode, rqname, 0, "현재가")
            self.current_price = current_price.strip()
            self.loop.quit()

    def update_price(self):
        price = self.get_current_price("005930")  # 삼성전자 종목 코드
        self.lineEdit.setText(str(price))
        self.update_profit()

    def execute_order(self):
        current_price = int(self.lineEdit.text())
        quantity = 100000 // current_price  # 10만 원 가치의 주식 매수
        self.kiwoom.send_order("삼성전자 매수", "0101", "0000", 1, "005930", quantity, current_price, "00", "")
        self.entry_price = current_price
        self.position_size = quantity

    def update_profit(self):
        if hasattr(self, 'entry_price'):
            current_price = int(self.lineEdit.text())
            profit = (current_price - self.entry_price) * self.position_size
            self.lineEdit_2.setText(f"{profit}원")
            self.check_auto_sell(current_price)

    def check_auto_sell(self, current_price):
        target_price = int(self.lineEdit_3.text())
        stop_loss_price = int(self.lineEdit_4.text())
        if current_price >= target_price or current_price <= stop_loss_price:
            self.kiwoom.send_order("삼성전자 매도", "0101", "0000", 2, "005930", self.position_size, current_price, "00", "")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StockTradingSystem()
    window.show()
    sys.exit(app.exec_())
