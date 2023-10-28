import sys
import os
  
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)


import torch
from project.module.models.swin_transformer import SwinTransformer

model = SwinTransformer()

x = torch.randn(4, 3, 224, 224)  # b, c, h, w, d, t

z = model(x)
loss = z.sum()
loss.backward()
print('done')
