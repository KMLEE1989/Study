import os

from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from config.errorCode import *
from PyQt5.QtTest import *
from config.kiwoomType import *
import sys

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()

        print("Kiwoom 클래스 입니다.")

        self.realType = RealType()

        ###########event loop 모음
        self.login_event_loop = None
        self.detail_account_info_event_loop = QEventLoop()
        self.calculator_event_loop = QEventLoop()
        ################################

        ############스크린 번호 모음
        self.screen_my_info = '2000'
        self.screen_calculation_stock = '4000'
        self.screen_real_stock = "5000" #종목별로 할당할 스크린 번호
        self.screen_meme_stock = "6000"
        self.screen_start_stop_real = "1000"

        ################################변수 모음
        self.account_num = None
        ################################

        #########################계좌 관련 변수
        self.use_money = 0
        self.use_money_percent = 0.5
        ########################################

        ################변수 모음
        self.portfolio_stock_dict = {}
        self.account_stock_dict = {}
        self.not_account_stock_dict = {}
        ##########################################

        ############################# 종목 분석 용
        self.calcul_data = []
        ######################################

        self.get_ocx_instance()
        self.event_slots()
        self.real_event_slots()

        self.signal_login_commConnect()
        self.get_account_info()
        self.detail_account_info() # 예수금 가져오는 것!
        self.detail_account_mystock() # 계좌평가 잔고 내역 요청
        self.not_concluded_account() #미체결 요청

        self.read_code() #저장된 종목들 불러온다
        self.screen_number_setting() #스크린 번호를 할당

        self.dynamicCall("SetRealReg(QString,QString,QString,QString)", self.screen_start_stop_real, '', self.realType.REALTYPE['장시작시간']['장운영구분'], "0")

        for code in self.portfolio_stock_dict.keys():
            screen_num = self.portfolio_stock_dict[code]['스크린번호']
            fids = self.realType.REALTYPE['주식체결']['체결시간']

            self.dynamicCall("SetRealReg(QString,QString,QString,QString)", screen_num, code,
                             fids, "1")
            print("실시간 등록 코드: %s, 스크린번호: %s, fid번호: %s" % (code, screen_num, fids))

    def get_ocx_instance(self):
        self.setControl('KHOPENAPI.KHOpenAPICtrl.1')

    def event_slots(self):
        self.OnEventConnect.connect(self.login_slot)
        self.OnReceiveTrData.connect(self.trdata_slot)

    def real_event_slots(self):
        self.OnReceiveRealData.connect(self.realdata_slot)

    def signal_login_commConnect(self):
        self.dynamicCall("CommConnect()")

        self.login_event_loop = QEventLoop()
        self.login_event_loop.exec_()

    def login_slot(self, errCode):
        print(errors(errCode))

        self.login_event_loop.exit()

    def get_account_info(self):
        account_list = self.dynamicCall('GetLoginInfo(String)', 'ACCNO')

        self.account_num = account_list.split(";")[0]

        print('나의 보유 계좌번호 %s ' % self.account_num)  #8018605611

    def detail_account_info(self):
        print('예수금을 요청하는 부분')

        self.dynamicCall('SetInputValue(QString, QString)','계좌번호', self.account_num)
        self.dynamicCall('SetInputValue(QString, QString)', '비밀번호', '0000')
        self.dynamicCall('SetInputValue(QString, QString)', '비밀번호입력매체구분 ', '00')
        self.dynamicCall('SetInputValue(QString, QString)', '조회구분', '2')
        self.dynamicCall('CommRqData(QString, QString, int, QString)','예수금상세현황요청','opw00001', '0', self.screen_my_info)


        self.detail_account_info_event_loop.exec_()

    def detail_account_mystock(self, sPrevNext='0'):
        print('계좌평가 잔고내역 요청하기 연속조회 %s' % sPrevNext)

        self.dynamicCall('SetInputValue(QString, QString)', '계좌번호', self.account_num)
        self.dynamicCall('SetInputValue(QString, QString)', '비밀번호', '0000')
        self.dynamicCall('SetInputValue(QString, QString)', '비밀번호입력매체구분 ', '00')
        self.dynamicCall('SetInputValue(QString, QString)', '조회구분', '2')
        self.dynamicCall('CommRqData(QString, QString, int, QString)', '계좌평가잔고내역요청', 'opw00018', sPrevNext, self.screen_my_info)

        self.detail_account_info_event_loop.exec_()

    def not_concluded_account(self, sPrevNext='0'):
        print('미체결 요청')

        self.dynamicCall('SetInputValue(QString, QString)', '계좌번호', self.account_num)
        self.dynamicCall('SetInputValue(QString, QString)', '체결구분', '1')
        self.dynamicCall('SetInputValue(QString, QString)', '매매구분', '0')
        self.dynamicCall('CommRqData(QString, QString, int, QString)', '실시간미체결요청', 'opt10075', sPrevNext, self.screen_my_info)

        self.detail_account_info_event_loop.exec_()

    def trdata_slot(self, sScrNo, sRQName, sTrCode, sRecordName, sPrevNext):

        if sRQName == '예수금상세현황요청':
            deposit = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, '예수금')

            print('예수금 %s' % type(deposit))
            print('예수금 형변환 %s' % int(deposit))

            self.use_money = int(deposit) * self.use_money_percent
            self.use_money = self.use_money / 4

            ok_deposit = self.dynamicCall('GetCommData(QString, QString, int, QString)', sTrCode, sRQName, 0, "출금가능금액")
            print('출금가능금액 %s' % ok_deposit)
            print('출금가능금액 형변환 %s' % int(ok_deposit))

            self.detail_account_info_event_loop.exit()


        elif sRQName == '계좌평가잔고내역요청':

            total_buy_money = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, '총매입금액')
            total_buy_money_result = int(total_buy_money)

            print('총매입금액 %s' % total_buy_money_result)

            total_profit_loss_money = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, "총평가손익금액")  # 출력 : 000000000009761
            total_profit_loss_money = int(total_profit_loss_money)

            print('총평가손익급액 %s' % total_profit_loss_money)

            total_profit_loss_rate = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, '총수익률(%)')
            total_profit_loss_rate_result = float(total_profit_loss_rate)

            print('총수익률(%%) : %s' % '%', total_profit_loss_rate_result)

            rows = self.dynamicCall('GetRepeatCnt(QString, QString)', sTrCode, sRQName)

            cnt = 0

            for i in range(rows):
                code = self.dynamicCall('GetCommData(QString, QString, int, QString)', sTrCode, sRQName, i, '종목번호')

                code = code.strip()[1:]

                code_nm = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "종목명")
                stock_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "보유수량")
                buy_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매입가")
                learn_rate = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "수익률(%)")
                current_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "현재가")
                total_chegual_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매입금액")
                possible_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매매가능수량")

                print("종목번호: %s - 종목명: %s - 보유수량: %s - 매입가:%s - 수익률: %s - 현재가: %s - 매입금액: %s - 매매가능수량: %s" % (code, code_nm, stock_quantity, buy_price, learn_rate, current_price, total_chegual_price, possible_quantity))

                if code in self.account_stock_dict:
                    pass
                else:
                    self.account_stock_dict[code] = {}

                code_nm = code_nm.strip()
                stock_quantity = int(stock_quantity.strip())
                buy_price = int(buy_price.strip())
                learn_rate = float(learn_rate.strip())
                current_price = int(current_price.strip())
                total_chegual_price = int(total_chegual_price.strip())
                possible_quantity = int(possible_quantity.strip())

                self.account_stock_dict[code].update({"종목명": code_nm})
                self.account_stock_dict[code].update({"보유수량": stock_quantity})
                self.account_stock_dict[code].update({"매입가": buy_price})
                self.account_stock_dict[code].update({"수익률(%)": learn_rate})
                self.account_stock_dict[code].update({"현재가": current_price})
                self.account_stock_dict[code].update({"매입금액": total_chegual_price})
                self.account_stock_dict[code].update({"매매가능수량": possible_quantity})

                cnt += 1

            # print("sPreNext : %s" % sPrevNext)
            print("계좌에 가지고 있는 종목 카운트 %s " % rows)

            if sPrevNext == '2':
                self.detail_account_mystock(sPrevNext="2")
            else:
                self.detail_account_info_event_loop.exit()

        elif sRQName == '실시간미체결요청':

            rows = self.dynamicCall('GetRepeatCnt(QString, QString)', sTrCode, sRQName)

            for i in range(rows):
                code = self.dynamicCall('GetCommData(QString, QString, int, QString)', sTrCode, sRQName, i, '종목번호')
                code_nm = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "종목명")
                order_no = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "주문번호")
                order_status = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "주문상태")  # 접수,확인,체결
                order_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "주문수량")
                order_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "주문가격")
                order_gubun = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "주문구분")  # -매도, +매수, -매도정정, +매수정정
                not_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "미체결수량")
                ok_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "체결량")

                code = code.strip()
                code_nm = code_nm.strip()
                order_no = int(order_no.strip())
                order_status = order_status.strip()
                order_quantity = int(order_quantity.strip())
                order_price = int(order_price.strip())
                order_gubun = order_gubun.strip().lstrip('+').lstrip('-')
                not_quantity = int(not_quantity.strip())
                ok_quantity = int(ok_quantity.strip())

                if order_no in self.not_account_stock_dict:
                    pass
                else:
                    self.not_account_stock_dict[order_no] = {}

                self.not_account_stock_dict[order_no].update({'종목코드': code})
                self.not_account_stock_dict[order_no].update({'종목명': code_nm})
                self.not_account_stock_dict[order_no].update({'주문번호': order_no})
                self.not_account_stock_dict[order_no].update({'주문상태': order_status})
                self.not_account_stock_dict[order_no].update({'주문수량': order_quantity})
                self.not_account_stock_dict[order_no].update({'주문가격': order_price})
                self.not_account_stock_dict[order_no].update({'주문구분': order_gubun})
                self.not_account_stock_dict[order_no].update({'미체결수량': not_quantity})
                self.not_account_stock_dict[order_no].update({'체결량': ok_quantity})

                print('미체결 종목 : %s ' % self.not_account_stock_dict[order_no])

            self.detail_account_info_event_loop.exit()

        elif '주식일봉차트조회'== sRQName:

            code = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, "종목코드")
            code = code.strip()
            print('%s 일봉데이터 요청' % code)

            cnt = self.dynamicCall('GetRepeatCnt(QString, QString)', sTrCode, sRQName)
            print('데이터 일수 %s' % cnt)

            for i in range(cnt):
                data = []

                current_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "현재가")  # 출력 : 000070
                value = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "거래량")  # 출력 : 000070
                trading_value = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "거래대금")  # 출력 : 000070
                date = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "일자")  # 출력 : 000070
                start_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "시가")  # 출력 : 000070
                high_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "고가")  # 출력 : 000070
                low_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "저가")  # 출력 : 000070

                data.append("")
                data.append(current_price.strip())
                data.append(value.strip())
                data.append(trading_value.strip())
                data.append(date.strip())
                data.append(start_price.strip())
                data.append(high_price.strip())
                data.append(low_price.strip())
                data.append("")

                self.calcul_data.append(data.copy())

            print(len(self.calcul_data))

            if sPrevNext == '2':
                self.day_kiwoom_db(code=code, sPrevNext=sPrevNext)

            else:

                print('총 일수 %s' % len(self.calcul_data))

                pass_success = False

                if self.calcul_data == None or len(self.calcul_data) < 120:
                    pass_success = False

                else:
                    total_price = 0
                    for value in self.calcul_data[:120]:
                        total_price += int(value[1])

                    moving_average_price = total_price / 120

                    #오늘자 주가가 120일 이평선에 걸쳐있는지 확인
                    bottom_stock_price = False
                    check_price = None

                    if int(self.calcul_data[0][7]) <= moving_average_price and moving_average_price <= int(self.calcul_data[0][6]):
                        print('오늘 주가 120이평선에 걸쳐있는 것 확인')
                        bottom_stock_price = True
                        check_price = int(self.calcul_data[0][6])

                    # 과거 일봉들이 120일 이평선보다 밑에 있는지 확인.
                    # 그렇게 확인을 하다가 일봉이 120일 이평선보다 위에 있으면 계산 진행
                    prev_price = None #과거의 일봉 저가
                    if bottom_stock_price == True:

                        moving_average_price_prev = 0
                        price_top_moving = False
                        idx = 1
                        while True:

                            if len(self.calcul_data[idx:]) < 120: #120일치가 있는지 계속 확인
                                print('120일치가 없음!')
                                break

                            total_price = 0
                            for value in self.calcul_data[idx:120+idx]:
                                total_price += int(value[1])
                            moving_average_price_prev = total_price / 120

                            if moving_average_price_prev <= int(self.calcul_data[idx][6]) and idx <= 5:
                                print('5일 동안 주가가 120일 이평선과 같거나 위에 있으면 조건 통과 못함')
                                price_top_moving = False
                                break

                            elif int(self.calcul_data[idx][7]) > moving_average_price_prev and idx > 5:
                                print('120일 이평선 위에 있는 일봉 확인됨')
                                price_top_moving = True
                                prev_price = int(self.calcul_data[idx][7])
                                break

                            idx += 1
                        #해당 부분 이평선이 가장 최근 일자의 이평선 가격보다 낮은지 확인
                        if price_top_moving == True:
                            if moving_average_price > moving_average_price_prev and check_price > prev_price:
                                print('포착된 이평선의 가격이 오늘자(최근일자) 이편선 가격보다 낮은 것 확인됨')
                                print('포착된 부분의 일봉 저가가 오늘자 일봉의 고가보다 낮은지 확인됨')
                                pass_success = True

                if pass_success == True:
                    print('조건부 통과됨')

                    code_nm = self.dynamicCall('GetMasterCodeName(QString)', code)

                    f = open('C:/Users/bit/PycharmProjects/pythonProject/files/condition_stock.txt', 'a', encoding='utf8')
                    f.write('%s\t%s\t%s\n' % (code, code_nm, str(self.calcul_data[0][1])))
                    f.close()

                elif pass_success == False:
                    print('조건부 통과 못함')

                self.calcul_data.clear()
                self.calculator_event_loop.exit()


    def get_code_list_by_market(self, market_code):

        code_list = self.dynamicCall('GetCodeListByMarket(QString)', market_code)
        code_list = code_list.split(';')[:-1]

        return code_list

    def calculator_fnc(self):

        code_list = self.get_code_list_by_market('10')
        print('코스닥 갯수 %s' % len(code_list))

        for idx, code in enumerate(code_list):

            self.dynamicCall('DisconnectRealData(QString)', self.screen_calculation_stock)

            print('%s / %s: KOSDAQ Stock Code : %s is updating....' % (idx+1, len(code_list), code))

            self.day_kiwoom_db(code=code)

    def day_kiwoom_db(self, code=None, date=None, sPrevNext='0'):

        QTest.qWait(3600)


        self.dynamicCall('SetInputValue(QString, QString)', '종목코드', code)
        self.dynamicCall('SetInputValue(QString, QString)', '수정주가구분', '1')

        if date != None:
            self.dynamicCall('SetInputValue(QString, QString)', '기준일자', date)

        self.dynamicCall('CommRqData(QString, QString, int, QString)', '주식일봉차트조회', 'opt10081', sPrevNext, self.screen_calculation_stock)

        self.calculator_event_loop.exec_()

    def read_code(self):

        if os.path.exists("C:/Users/bit/PycharmProjects/pythonProject/files/condition_stock.txt"):
            f = open("C:/Users/bit/PycharmProjects/pythonProject/files/condition_stock.txt", "r", encoding="UTF-8")

            lines = f.readlines()
            for line in lines:
                if line != "":
                    ls = line.split("\t")

                    stock_code = ls[0]
                    stock_name = ls[1]
                    stock_price = int(ls[2].split("\n")[0])
                    stock_price = abs(stock_price)

                    self.portfolio_stock_dict.update({stock_code:{"종목명":stock_name, "현재가":stock_price}})

            f.close()

            print(self.portfolio_stock_dict)

    def screen_number_setting(self):

        screen_overwrite = []

        #계좌평가잔고내역에 있는 종목들

        for code in self.account_stock_dict.keys():
            if code not in screen_overwrite:
                screen_overwrite.append(code)

        #미체결에 있는 종목들
        for order_number in self.not_account_stock_dict.keys():
            code = self.not_account_stock_dict[order_number]['종목코드']

            if code not in screen_overwrite:
                screen_overwrite.append(code)

        #포트폴리오에 담겨있는 종목들

        for code in self.portfolio_stock_dict.keys():
            if code not in screen_overwrite:
                screen_overwrite.append(code)


        #스크린번호 할당

        cnt = 0
        for code in screen_overwrite:

            temp_screen = int(self.screen_real_stock)
            meme_screen = int(self.screen_meme_stock)

            if (cnt % 50) == 0:
                temp_screen += 1
                self.screen_real_stock = str(temp_screen)

            if (cnt % 50) == 0:
                meme_screen += 1
                self.screen_meme_stock = str(meme_screen)

            if code in self.portfolio_stock_dict.keys():
                self.portfolio_stock_dict[code].update({"스크린번호": str(self.screen_real_stock)})
                self.portfolio_stock_dict[code].update({"주문용스크린번호":str(self.screen_meme_stock)})

            elif code not in self.portfolio_stock_dict.keys():
                self.portfolio_stock_dict.update({code: {"스크린번호":str(self.screen_real_stock), "주문용스크린번호":str(self.screen_meme_stock)}})

            cnt += 1

        print(self.portfolio_stock_dict)

    def realdata_slot(self, sCode, sRealType, sRealData):

        if sRealType == "장시작시간":
            fid = self.realType.REALTYPE[sRealType]['장운영구분']
            value = self.dynamicCall("GetCommRealData(QString, int)", sCode, fid)

            if value == '0':
                print("장 시작 전")

            elif value == '3':
                print("장 시작")

            elif value == "2":
                print("장 종료, 동시호가로 넘어감")

            elif value == "4":
                print("3시 30분 장 종료")

        elif sRealType == "주식체결":
            print(sCode)












