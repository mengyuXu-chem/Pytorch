# import package
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset   #dataset是一个抽象类，无法实例化，只能被继承
from torch.utils.data import DataLoader

#使用Minibatch进行训练

xy = np.loadtxt('xyz.csv.gz',delimiter=',',dtype=np.float32)  #delimiter为分隔符   dtype为数据类型
x_data = torch.from_numpy(xy[:,:-1])   #取除最后一列得其他列
y_data = torch.from_numpy(xy[:,[-1]])  #只取最后一列作为y值  创建得两个Tensor

class MiniBatch(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])


    def __getitem__(self,index):  #通过索引获取值
        return self.x_data[index], self.y_data[index]

    def __len__(self):   #获取长度
        return self.len
    
dataset = MiniBatch('filepath')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)  #num_workers 设置多线程，增加读取效率

class MiniBatch(torch.nn.Module):
    def __init__(self):
        super(MiniBatch,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid(8,6)    #更改激活函数
    def forwark(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = MiniBatch()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


#若windows系统下无法进行多线程，则
# if __name__ == '__main__':
for epoch in range(100):
    for i,data in enumerate(train_loader,0):  #enumerate可以获得当前迭代次数
        #prepare data
        inputs, labels = data
        #forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        #backward
        optimizer.zero_grad()
        loss.backward()
        #Update
        optimizer.step()


