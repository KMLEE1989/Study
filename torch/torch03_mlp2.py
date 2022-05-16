# gpu
# criterion 이슈

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch :  1.11.0  사용DEVICE : cuda

#1.데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],            #행과 열을 바꾸기 위해 utilize transpose method 
            [1,1.1,1.2,1.3,1.4,1.5,
             1.6,1.5,1.4,1.3],
            [10,9,8,7,6,5,4,3,2,1]]) 
y = np.array([11,12,13,14,15,16,17,18,19,20])

x = np.transpose(x)

x = torch.FloatTensor(x).to(DEVICE)  #
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)  #(10,) ->(10,1)

print(x,y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape, y.shape) # torch.Size([10, 3]) torch.Size([10, 1])

#2. 모델 구성
# model = Sequential()   #S는 대문자 그래서 클래서    시퀀셜이라는 클래스를 모델로 정의 하겠다
# model.add(Dense(1, input_dim=1))
#model = nn.Linear(1, 1).to(DEVICE)     #인풋, 아웃풋

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 4),
    nn.Linear(4, 2),
    nn.Linear(2, 1),
).to(DEVICE)

#3. 컴파일,훈련
#model.compile(loss='mse', optimizer='adam')  #그은선과 실제 데이터의 거리 loss  민스쿼드에라 평균 제곱(하는 이유 양수화 음수가 나올수 없다) 에라 (작으면 좋다)    옵티마이저(엠에스이를 최소화 )

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
print(optimizer)

#model.fit(x, y, epochs=100, batch_size=1)
def train(model, criterion, optimizer, x, y):
    #model.train()  #훈련모드
    optimizer.zero_grad()  #기울기 초기화
    
    hypothesis = model(x)

    loss = criterion(hypothesis, y)
    #loss = nn.MSELoss()(hypothesis, y) 에러
    #loss = nn.MSELoss()(hypothesis, y)
    #loss = F.mse_loss(hypothesis, y)
    
    loss.backward()  #기울기값 계산까지다
    optimizer.step()  #가중치 수정
    return loss.item()

epochs = 10000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))

print('===============================================================')

#4. 평가,예측
#loss = model.evaluate(x,y)  #최소의 로스가 될 준비 
def evaluate(model, criterion, x, y):
    model.eval()  #평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)    
print('최종 loss : ', loss2)


result = model(torch.Tensor([[10, 1.3, 1]]).to(DEVICE))
print('10, 1.3, 1의 예측값: ', result.item())

# 최종 loss :  4.7530298616038635e-05
# 4의 예측값:  3.9854846000671387

