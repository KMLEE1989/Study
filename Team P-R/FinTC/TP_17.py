#!/usr/bin/python
# -*- coding: utf-8 -*-

#class 기초
class B_school():
    def __init__(self):
        print("B 클래스 입니다.")

    def english(self):

        student_name = '원빈'

        return student_name

class A_school():
    def __init__(self):
        print('A 클래스 입니다.')

        bb=B_school()
        student_name = bb. english()

        print(self.student_name)

A_school()




