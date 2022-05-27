from sklearn.datasets import load_breast_cancer, load_boston
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch :  1.11.0  사용DEVICE : cuda

#1. 데이터

# datasets = load_breast_cancer()
datasets = load_boston()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)
# (398, 30) torch.Size([398, 1])

print(type(x_train),type(y_train))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train) # x, y 합친다. 
test_set = TensorDataset(x_test, y_test)

print(len(train_set)) # 354
print(type(train_set)) 
print(train_set[0]) 

train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
test_loader = DataLoader(test_set, batch_size=36, shuffle=False)

#2. 모델

# model = nn.Sequential(
#     nn.Linear(30, 32),
#     nn.ReLU(),
#     nn.Linear(32, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid()
  
# ).to(DEVICE)

class Model(nn.Module): 
    def __init__(self, input_dim, output_dim):
        #super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        out = self.linear4(x)
        return out
    
model = Model(13, 1).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    #model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
    
    
        loss.backward()  #역전파
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)
    
EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epoch: {}, loss: {:.6f}'.format(epoch, loss))

print('=========================== 평가, 예측 ===============================')
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0 
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
            
    return total_loss
        
loss = evaluate(model, criterion, test_loader)
print('loss : ', loss)

# y_predict = (model(x_test) >= 0.5).float()
# print(y_predict[:10])
y_predict = model(x_test)

# score = (y_predict == y_test).float().mean()
# print('accuracy : {:.4f}'.format(score))


from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# score = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())

# print('accuracy : {:.4f}'.format(score))

score = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print('r2_score : {:.4f}'.format(score))

# =========================== 평가, 예측 ===============================
# loss :  69.74465084075928
# r2_score : 0.8090