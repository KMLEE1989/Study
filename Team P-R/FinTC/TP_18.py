#!/usr/bin/python
# -*- coding: utf-8 -*-

#class 기초
# class B_school():
#     def __init__(self):
#         print("B 클래스 입니다.")
#
#     def english(self):
#
#         student_name = '원빈'
#
#         return student_name
#
# class A_school():
#     def __init__(self):
#         print('A 클래스 입니다.')
#
#         bb=B_school()
#         student_name = bb. english()
#
#         print(self.student_name)
#
# A_school()

#상속
class Parent():
    def __init__(self):
        print("부모 클래스!")

        self.money = 500000000000000000

    def book(self):
        print('부모의 서재입니다.')

class Child_1(Parent):
    def __init__(self):
        super().__init__()
        print("첫번째 자식입니다.")
        print(self.money)

class Child_2(Parent):
    def __init__(self):
        print('두번째 자식입니다.')
        self.book()



Child_1()
Child_2()




