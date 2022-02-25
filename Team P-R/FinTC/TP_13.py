#!/usr/bin/python
# -*- coding: utf-8 -*-

#1 - 함수 만들기
def english():
    print('영어과 입니다.')
    print('영어과에서 영어 공부 합니다')

english()

#2 - 해당 함수에 인자 전달.
def math(name):
    print('수학과 입니다.')
    print('%s 학생이 전학 왔습니다.' % name)
math('원빈')

# 영어과 입니다.
# 영어과에서 영어 공부 합니다
# 수학과 입니다.
# 원빈 학생이 전학 왔습니다.
