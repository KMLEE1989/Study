#1

a_tuple = ('키움증권', '대신증권', '네이버증권')

kiwoom_count = 0

for value in a_tuple:
    if value == "키움증권":
        kiwoom_count += 1

print("키움종목 매수량 : %s" % kiwoom_count)


