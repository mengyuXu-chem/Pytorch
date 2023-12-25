# import package
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F   # using ReLu rather than sigamod 
import torch.optim as optim  # optimizer

# Tensor need to Gaussian 
# 全连接神经网络

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),   # Convert the PIL image to Tensor
    transforms.Normalize((0.1307,),(0.3081,))   #0.1307 mean  0.3081 std  Normalize归一化
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               transform=transform,
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=False,
                               transform=transform,
                               download=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)  #shuffle close, convient view dataset

class ExamMNIST(torch.nn.Module):
    def __init__(self):
        super(ExamMNIST,self).__init__()
        self.L1 = torch.nn.Linear(784,512)
        self.L2 = torch.nn.Linear(512,256)
        self.L3 = torch.nn.Linear(256,128)
        self.L4 = torch.nn.Linear(128,64)
        self.L5 = torch.nn.Linear(64,10)  #10分类问题

    def forward(self,x):
        x = x.view(-1,784)   #原始数据转为N个样本*784的矩阵  784 = 28 * 28
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))   #前四层用RuLu函数激活
        return self.L5(x)     #最后一层使用softmax
    
model = ExamMNIST()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  #momentum为冲量，带冲量优化

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss /300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data,dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct /total))

#training 
if __name__ == '__main__':
    for epoch in range(10):
       train(epoch)
       test()

