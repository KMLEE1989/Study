import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch :  1.11.0  사용DEVICE : cuda
  

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_train=np.array([1,2,3,4,5,6,7])
y_test=np.array([8,9,10])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape) #torch.Size([7, 1]) torch.Size([7, 1])
print(x_test.size(), y_test.size()) #torch.Size([3, 1]) torch.Size([3, 1])

#2. 모델구성

model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.ReLU(),
    nn.Linear(3,4),
    nn.Linear(4,1)
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

print(optimizer)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis, y)
    
    
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
#4. 평가,예측
#loss = model.evaluate(x,y)  #최소의 로스가 될 준비 
def evaluate(model, criterion, x, y):
    model.eval()  #평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)    
print('최종 loss : ', loss2)

result = model(x_test.to(DEVICE))
print('x_test의 예측값 : ', result.cpu().detach().numpy()) # 요건 알아서 수정


