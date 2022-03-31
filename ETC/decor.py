def main_function():
    print("MAIN FUNCTION START")
    
    
# 추가적으로 문장을 출력하기 전과 후에 날짜와 시간을 출력하고 싶다면?

import datetime

def main_function():
    print(datetime.datetime.now())
    print("MAIN FUNCTION START")
    print(datetime.datetime.now())
    
# 매우 간단  -> 특별히 복잡하게 추가할게 없는 작업 
# 만약 이와 같은 패턴의 함수가 여러번 있다면?
    
import datetime

def main_function_1():
     print("MAIN FUNCTION 1 START")

def main_function_2():
     print("MAIN FUNCTION 2 START")

def main_function_3():
     print("MAIN FUNCTION 3 START")

#..... X 100번..

#여기에도 각 함수의 문장이 출력되기 전과 후에 시간을 출력하고 싶다면?

import datetime

def main_function_1():
     print(datetime.datetime.now())
     print("MAIN FUNCTION 1 START")
     print(datetime.datetime.now())

def main_function_2():
     print(datetime.datetime.now())
     print("MAIN FUNCTION 2 START")
     print(datetime.datetime.now())

def main_function_3():
     print(datetime.datetime.now())
     print("MAIN FUNCTION 3 START")
     print(datetime.datetime.now())

#.... X 100번

# 반복되는 구문이 많아짐 -> 소스가 지저분해지고, 본 main 함수의 가독성도 떨어짐

import datetime

def datetime_decorator(func):
        def decorated():
                print(datetime.datetime.now())
                func()
                print(datetime.datetime.now())
        return decorated

@datetime_decorator
def main_function_1():
        print("MAIN FUNCTION 1 START")

@datetime_decorator
def main_function_2():
        print("MAIN FUNCTION 2 START")

@datetime_decorator
def main_function_3():
        print("MAIN FUNCTION 3 START")

#..... X 100번

#decorator 함수를 재사용 -> main 함수에 대한 가독성과 직관성 좋아짐. @ 간편히 코드 작성

# decorator 선언된 부분:

# 먼저 decorator 역할을 하는 함수를 정의, 이 함수에서 decorator가 적용될 함수를 인자로 받음 
# python 은 함수의 인자로 다른 함수를 받을 수 있다는 특징을 이용 
# decorator 역할을 하는 함수 내부에 또 한번 함수를 선언(nested function)하여 
# 여기에 추가적인 작업(시간 출력) 을 선언해 주는 것이다. 
# nested 함수를 return 해주면 된다. 

#  마지막으로, main 함수들의 앞에 @를 붙여 decorator 역할을 하는 함수를 호출해 준다. 





