import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os


dataset = dset.ImageFolder( 
  root=os.getcwd()+"/data/",
  transform=transforms.Compose([ 
  transforms.ToTensor(), 
  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
 ]) 
) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2) 