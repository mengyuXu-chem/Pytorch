import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('xyz.csv.gz',delimiter=',',dtype=np.float32)  #delimiter为分隔符   dtype为数据类型
x_data = torch.from_numpy(xy[:,:-1])   #取除最后一列得其他列
y_data = torch.from_numpy(xy[:,[-1]])  #只取最后一列作为y值  创建得两个Tensor

class MultiDimensionInput(torch.nn.Module):
    def __init__(self):
        super(MultiDimensionInput,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid(8,6)    #更改激活函数
    def forwark(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = MultiDimensionInput()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):                                   ######training dataset
    #Forward
    y_pred = model(x_data)  
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())        
    #backward
    optimizer.zero_grad()
    loss.backward()   
    #update
    optimizer.step()  






