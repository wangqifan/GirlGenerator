import torch.nn as nn
import torch.nn.functional as F

"""
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.liner=nn.Sequential(
            nn.Linear(100,1500),
            nn.ReLU(True),
            nn.Linear(1500,2028),
            nn.ReLU(True)
        )
        self.conv1=nn.Sequential(
              nn.Conv2d(3,12,3,1,padding=1),
              nn.BatchNorm2d(12),
              nn.ReLU(True)
        )
        self.BasicBlock=BasicBlock(12,12)
        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(12,24,3,2,padding=1),
              nn.BatchNorm2d(24),
              nn.ReLU(True)
        )
        self.BasicBlock2=BasicBlock(24,24)
        self.conv3=nn.Sequential(
              nn.ConvTranspose2d(24,12,3,2,padding=1),
              nn.BatchNorm2d(12),
              nn.ReLU(True),
              nn.Conv2d(12,3,6,1),
              nn.BatchNorm2d(3),
              nn.Tanh()
        )
    def forward(self,x):
        x=self.liner(x)
        x=x.view(-1,3,26,26)
        x=self.conv1(x)
        x=self.BasicBlock(x)
        x=self.conv2(x)
        x=self.BasicBlock2(x)
        x=self.conv3(x)
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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
    
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            nn.Tanh() 
        )

    def forward(self, input):
        return self.main(input)
"""
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.liner=nn.Sequential(
            nn.Linear(100,1500),
            nn.ReLU(True),
            nn.Linear(1500,2028),
            nn.ReLU(True)
        )
        self.conv1=nn.Sequential(
              nn.Conv2d(3,64,3,1,padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(True)
        )
        self.BasicBlock=BasicBlock(64,64)
        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(64,128,3,2,padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(True)
        )
        self.BasicBlock2=BasicBlock(128,128)
        self.conv3=nn.Sequential(
              nn.ConvTranspose2d(128,64,3,2,padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(True),
              nn.Conv2d(64,3,6,1),
              nn.BatchNorm2d(3),
              nn.Tanh()
        )
    def forward(self,x):
        x=self.liner(x)
        x=x.view(-1,3,26,26)
        x=self.conv1(x)
        x=self.BasicBlock(x)
        x=self.conv2(x)
        x=self.BasicBlock2(x)
        x=self.conv3(x)
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