############################# 마음껏 튜닝 ㄱㄱ
x = 10            # 임의로 바꿔도 되.
y = 10            # 목표값
w = 1           # 가중치 초기값 
lr = 0.2
epochs = 500 

for i in range(epochs):
    predict = x * w
    loss = (predict-y)**2
    
    
    # 가중치와 epoch 도 너서 아래 print를 수정
    print("Loss : " , round(loss, 4), "\tPredict : ", round(predict, 4))
    
    up_predict = x*(w + lr)
    up_loss = (y-up_predict) ** 2
    
    down_predict = x*(w - lr)
    down_loss = (y-down_predict) ** 2
    
    
    if(up_loss > down_loss):
        w = w - lr
        
    else:
        w = w + lr
        
        