#相比于传统的全连接神经网络，对于图像处理，使用卷积神经网络，可以更抽象的提取空间特征
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F   # using ReLu rather than sigamod 
import torch.optim as optim  # optimizer

# Tensor need to Gaussian 
# 卷积神经网络

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

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)  

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

#如何使用GPU计算：需要将运算迁移至GPU
    '''
    1、将模型迁移至GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
    2、迁移至device
    model.to(device)
    3、将数据也迁移至GPU
    inputs, target = data ------------ inputs, target = inputs.to(device), target.to(device)  训练集以及测试集的都要放入与model一致的显卡中
    images, labels = data ------------ inputs, target = inputs.to(device), target.to(device)

    '''

    
model = CNN()
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