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

class AdvancedCNN(torch.nn.Module):  #通过高级CNN连接实现模块的大型运用
    def __init__(self,in_channels):
        super(AdvancedCNN,self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)           # number 1

        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)          # number 2
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)              # number 3
        self.branch3x3_2 = torch.nn.Conv2d(16, 24,kernel_size=3, padding= 1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24,kernel_size=3, padding= 1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)              # number 4

    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1,padding=1)   
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]   #对有所的分支加权
        return torch.cat(outputs,dim=1)

class ResidualBlock(torch.nn.Module):  #何凯明的跳连接可以避免梯度消失的问题
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)
    
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512,10)
    
    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.rblock1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size,-1)
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