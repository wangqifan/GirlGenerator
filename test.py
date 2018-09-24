import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms  as transforms 
from torchvision.utils import save_image



def to_img(x):
    out=(x+1)*0.5
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 96, 96)
    return out


z_dimension=100

generate=torch.load("G.pkl")

z=Variable(torch.randn(128,z_dimension)).cuda()
output=generate(z)
output=output.view(-1,3,96,96)
images=to_img(output)
images=images.cpu()
save_image(images.data,"temp.jpg")