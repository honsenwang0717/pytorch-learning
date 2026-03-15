import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_CNN(nn.Module):
    def __init__(self): #初始化方法,其中self类似Java的this,代表当前对象
        '''
            super()是用来调用父类，然后通过.init()调用父类的初始化方法，必须写，因为继承了nn.Module类，所以必须主动调用父类的初始化方法，具体调用流程如下：
            1 MyModel.__init__()
            2 super().__init__()
            3 nn.Module.__init__()
            4 初始化 PyTorch 模块系统
            5 创建 self.fc1 和 self.lstm 等属性
        '''
        super().__init__()  

        # 卷积层,输入为一张彩色图片，所有是3层，输出16个通道，卷积核大小为3x3，步长为1，填充为1，高宽不变
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1) #输入通道数，输出通道数，卷积核大小，步长，填充
        # 卷积层，输入为16个通道，输出32个通道，卷积核大小为3x3，步长为1，填充为1，高宽不变
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        #池化层，池化核大小为2x2，步长为2，高宽减半
        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2) 

        #用一个全连接层作为最终的输出，模拟一个图像分类的任务，输入是最终图像的拉伸成一维向量的长度，输出是10，代表10个类别。
        #注意：输出、输出需要根据实际情况具体计算。根据前面的计算，每次卷积高宽不变，但是通道数变了

        #希望有个过度，所以用了两个全连接层，或者说，中间有个隐藏层。
        self.fc = nn.Linear(32*12*12,100) 
        #经验的做法，理论上也可以只有一个全连接层，但是那样的话，中间的计算量会很大，所以用了两个全连接层。
        self.fcc = nn.Linear(100,7)

        #还有一些其他的可能用到的层，比如dropout，全连接层等
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*8*8,10) #理论上，目前只是一个线形层，输入是32*8*8，输出是10，因为缺少了激活函数。但是现在约定俗成，还是叫他“全连接层”，需要用到激活函数的时候再处理就行
        self.lstm = nn.LSTM(input_size=32*8*8,hidden_size=128,num_layers=2,batch_first=True)
        self.embedding = nn.Embedding(num_embeddings=10000,embedding_dim=32)

        self.test = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32*12*12,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxPool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxPool(x)
        x = x.view(-1,32*12*12) #将x展平，-1表示自动计算维度，32*12*12是展平后的维度。这个是展平操作，将多维度的tensor展平成一维的tensor。 还可以用x = torch.flatten(x,1)或者x=x.flatten(1)来展平，1表示从第1维度开始展平。
        x = F.relu(self.fc(x)) #因为是线性的转化，所以需要用激活函数来处理。
        x = self.fcc(x) # 输出层不加激活函数，CrossEntropyLoss 内部自带 Softmax

        return x

