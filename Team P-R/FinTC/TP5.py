#
# stock_name = '네이버증권'
#
# if stock_name == '대신증권' or stock_name == '키움증권':
#     print("대신증권이고 또는 키움증권 이네요!")
#
# elif stock_name == "네이버증권" or stock_name == "카카오 증권":
#     print("네이버증권 또는 카카오증권 이네요!")
#
# # 네이버증권 또는 카카오증권 이네요!

#and
# stock_price =1000
#
# if stock_price < 900 and stock_price > 2000:
#     print("주식 가격이 900원 보다 작거나! 200원 보다 크네요!")
#
# elif stock_price > 500 and stock_price < 2000 :
#     print("500원 보다 크고 2000원 보다 작다!")
#
# else:
#     print("포함되는 게 없네?")
# 500원 보다 크고 2000원 보다 작다!
#in

# stock_name = "삼성전자"
#
# if stock_name in ['LG전자', '애플', 'MS', '삼성전자']:
#     print("삼성전자가 존재한다!")
#
# 삼성전자가 존재한다!

stock_name = '삼성전자'
if stock_name in ['LG전자', '애플', 'MS', '삼성전자'] \
        and (stock_name == "홈즈" or stock_name == "삼성전자")\
            and stock_name != '가디언':
    print("리스트에 포함되어 있으면서")
    print("회사 이름은 홈즈 이거나 삼성전자 이다!")
    print("가디언과는 같지 않냐")

