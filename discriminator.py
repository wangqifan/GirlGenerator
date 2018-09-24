import torch.nn as  nn
import torch.nn.functional as F

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,4),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        self.BasicBlock=BasicBlock(32,32)
        self.dis2=nn.Sequential(
            nn.Conv2d(32,64,5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,5),
            nn.ReLU(),
            nn.Conv2d(128,256,3),
            nn.ReLU()
        )
        self.full=nn.Sequential(
            nn.Linear(256*3*3,640),
            nn.Linear(640,2)
        )
    def forward(self,x):
        x=self.dis(x)
        x=self.BasicBlock(x)
        x=self.dis2(x)
        x=x.view(-1,256*3*3)
        x=self.full(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self,in_planes,planes):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=x
        out=F.relu(out)
        return out  
