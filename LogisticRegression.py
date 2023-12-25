import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])  #Tensor type
y_data = torch.Tensor([[0],[0],[1]])   #from LinearRegression        ######data prepare 


class LogisticRegressionModel(torch.nn.Module):  # nn neural network              ######Design model using Class
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()   # 继承父类构造器
        self.linear = torch.nn.Linear(1,1)   # 构造函数，构造线性回归函数  Linear（权重+偏置）

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred           # 正向传播，前馈机制，为了得到loss函数   
    
    # no backward, include in torch.nn.Module.

model = LogisticRegressionModel()

#  document: class torch.nn.Linear(in_features,out_features,bias=True)
#  in_features  number dimension of input, x
#  out_features number dimension of output, y 

criterion = torch.nn.BCELoss(size_average=False)  #评判标准                       ######评判标准以及优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #优化器，优化更新权重以及偏置  lr = learning rate

for epoch in range(100):                                    ######training dataset
    y_pred = model(x_data)  
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())        
    
    optimizer.zero_grad()
    loss.backward()   
    optimizer.step()  #update

print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred = ', y_test.data)

x = np.linspace(0,20,200)
x_t = torch.Tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('hours')
plt.ylabel('probability of pass')
plt.grid()
plt.show()