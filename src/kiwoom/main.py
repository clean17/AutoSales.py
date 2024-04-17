from pykiwoom.kiwoom import *
import pprint

class MyKiwoom(Kiwoom):
    def __init__(self):
        super().__init__()
        self.connect_event_slots()

    def connect_event_slots(self):
        self.OnReceiveMsg.connect(self.on_receive_msg)
        self.OnReceiveChejanData.connect(self.on_receive_chejan_data)

    def on_receive_msg(self, screen_no, rqname, trcode, msg):
        print(f"화면번호: {screen_no}, 요청이름: {rqname}, TR코드: {trcode}, 메시지: {msg}")

    def on_receive_chejan_data(self, gubun, item_cnt, fid_list):
        주문번호 = self.GetChejanData(9203)
        print(f"주문 번호: {주문번호}, 구분: {gubun}, 체결 정보: {item_cnt}")

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)

state = kiwoom.GetConnectState()
if state == 0:
    print("미연결")
elif state == 1:
    print("연결완료")

전일가 = kiwoom.GetMasterLastPrice("005930")
print(int(전일가))

# 종목 리스트 출력
#group = kiwoom.GetThemeGroupList(1)
#pprint.pprint(group)

# 주문 정보 설정
account_number = kiwoom.GetLoginInfo("ACCNO")[0]   # 계좌번호
if (not account_number):
    print('계좌정보를 찾을 수 없습니다.')
    sys.exit(1)  # 프로그램 종료, 에러 코드 1
    
order_type = 1   # 신규매수
code = "005930"  # 삼성전자 종목코드
quantity = 1     # 매수할 주식 수량
price = "70000"  # 주문 가격
hoga = "00"      # 지정가

# 10만원 가치에 해당하는 주식 수량 계산
# 주의: 실제 사용시에는 현재 가격을 조회하여 계산해야 합니다.
market_price = 80000  # 현재 삼성전자 주식 가격 예시
total_amount = 100000  # 총 매수하고자 하는 금액
quantity = total_amount // market_price  # 매수할 주식 수량 계산

# 주식 매수 주문
kiwoom.SendOrder("주식매수", "0101", account_number, order_type, code, quantity, price, hoga, "")

mykiwoom = MyKiwoom()
mykiwoom.on_receive_chejan_data()

# print('매수주문이 체결되었습니다.')