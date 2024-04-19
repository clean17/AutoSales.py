import sys
from PyQt5.QtWidgets import QApplication
from pykiwoom.kiwoom import Kiwoom

class StockTradingSystem:
    def __init__(self):
        self.kiwoom = Kiwoom()
        self.kiwoom.CommConnect(block=True)  # 로그인

    def get_current_price(self, code):
        """ 종목 코드에 해당하는 현재가를 가져오는 함수 """
        # 데이터 요청
        self.kiwoom.SetInputValue("종목코드", code)
        self.kiwoom.CommRqData("opt10001_req", "opt10001", 0, "0101")

        # 데이터 추출
        price = self.kiwoom.GetCommData("opt10001", "opt10001_req", 0, "현재가")
        print(price)
        return 0
        # return abs(int(price.strip()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    system = StockTradingSystem()
    price = system.get_current_price("005930")
    print("삼성전자 현재가:", price)
    sys.exit(app.exec_())
