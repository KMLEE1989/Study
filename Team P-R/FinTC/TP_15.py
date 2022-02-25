#!/usr/bin/python
# -*- coding: utf-8 -*-

#1 - 함수 만들기
def english():
    print('영어과 입니다.')
    print('영어과에서 영어 공부 합니다')

#2 - 해당 함수에 인자 전달.
def math(name, eng):
    print('수학과 입니다.')
    eng()


#3 - 함수의 인자로 함수 전달
math(name='원빈', eng=english)

#4-Return !!!!!!!
def school():
    name = '광희'
    money = 5000

    return name

n = school()
print(n)

#5 샘플!

def condition(name=None, current=0):
    if current > 5000:
        print('종목이름: %s, 현재가: %s' % (name, current))

a_dict = {'종목명': "삼성", "현재가":50000}
condition(name='삼성', current=50000)

# 영어과 입니다.
# 영어과에서 영어 공부 합니다
# 수학과 입니다.
# 원빈 학생이 전학 왔습니다.


# 원빈, 고수, 김태희 학생이 전학 왔습니다.

