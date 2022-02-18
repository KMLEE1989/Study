a_list = ['키움증권', '대신증권', '네이버증권', '이베스트', '손오공']
b_list = ['오토', '구글', '페이스북', '애플', '오션']

for index, value in enumerate(b_list):
    a_list[index] = value

# print(a_list)
# ['오토', '구글', '페이스북', '애플', '오션']
# for value in b_list:
#     a_list.append(value)

# print(a_list)

# ['키움증권', '대신증권', '네이버증권', '이베스트', '손오공', '오토', '구글', '페이스북', '애플', '오션']



# print(type(a_list))
# <class 'list'>

# a_len = len(a_list)
# print(a_len)  #5

# a_list.append('카카오')
# print(a_list)
# ['키움증권', '대신증권', '네이버증권', '이베스트', '손오공', '카카오']

# del a_list[0]
#
# print(a_list)

# print(a_list[0])
#
# 키움증권

# for index, value in enumerate(a_list):
#     print('%s - %s' % (index, value))

# 0 - 키움증권
# 1 - 대신증권
# 2 - 네이버증권
# 3 - 이베스트
# 4 - 손오공

# cnt = 0
# for value in a_list:
#     print('%s - %s' % (cnt, value))
#
#     cnt += 1
# 0 - 키움증권
# 1 - 대신증권
# 2 - 네이버증권
# 3 - 이베스트
# 4 - 손오공

# a_list = ["키움증권", "대신증권", "네이버증권", "이베스트", "손오공"]
# check_stock = "키움증권"
#
# cnt = 0
# for value in a_list:
#     if value == check_stock:
#         del a_list[cnt]
#
#     cnt += 1
#
# print(a_list)
#
# ['대신증권', '네이버증권', '이베스트', '손오공']


