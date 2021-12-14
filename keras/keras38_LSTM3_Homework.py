from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model=Sequential()


# LSTM
# Forget Gate Layer: 어떠한 정보를 반영할지에 대한 결정
#                    Sigmoid 활성화 함수를 통해 0~1 사이의 값 출력
                     #여기서 Sigmoid는 Cell State에 어떤 정보를 버릴지 유지할기 졀정
# Input Gate Layer: 새로운 정보가 cell state에 저장이 될지 결정하는 게이트
#                   sigmoid layer, tanh layer로 구성
#                     Input Gate Layer: 어떤 값 갱신
#                     Tanh Layer: Cell State에 새로운 후보값 생성
#                     -> 최종 Cell State 갱신 값 생성
# # update Cell state: forget gate와 input gate에서 출력된 값들을 cell state로 업데이트

# Output Gate Layer: 출력값 결정 단계
          

# tanh: 0보다 값이 작을 수록 -1의 값을  0보다 값이 클 수록 1의 값을 출력한다. 평균적으로 나오게 되는 값은 0이된다.
# sigmoid나 relu와 달리 RNN구조가 필요한 유의미한 값의 범위를 유지해 줄 수 있기 때문에 tanh가 사용된다.

    #  - LSTM은 [0,1] 정규화 데이터로 학습

    #  - LSTM predict 결과는 [0,1] 정규화됨
    
#     LSTM 평가와 관련된 데이터 모양

#             : X_test (평가용 입력 데이터) ==> numpy 3차원 행렬

#             : Y_test (평가용 출력 데이터) ==> numpy 2차원 행렬

#             : Pred (LSTM 예측값) : numpy 2차원 행렬

#      - LSTM 예측 결과의 시각화

#             : Seaborn package의 lineplot 사용 예정

#             : Lineplot 함수는 1차원 행렬 요구

#      - Numpy flatten 함수

#             : Numpy n-차원 행렬을 1차원으로 축소

# 기본 개념
# RNN은 매번 스텝마다 과정을 반복, 역전파 시 더 많은 곱셈 연산에 따른 경사 감소로 뒤의 노드까지 영향 불가.
# 장기 의존성을 학습할 수 있는 RNN의 한 종류
# 순환 신경망의 장기 의존성 문제 해결하기 위해 셀 스테이트 기반 신경망 모델
# 여러 종류의 게이트가 잇어 입력을 선별적으로 허용, 계산 결과를 선별적으로 출력


# 구조!
# LSTM 핵심: 셀 스테이트 (Cell State)
# LSTM에 들어있는4개 상호작용 레이어가 있는 반복 모듈

#*시그모이드는 Cell State에 어떤 정보 버릴지 유지할 지 결정!*




