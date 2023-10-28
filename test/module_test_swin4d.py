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
from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D

model = SwinTransformer4D(
    img_size=(96, 96, 96, 20),
    in_chans=2,
    embed_dim=24,
    window_size=(4, 4, 4, 2),
    first_window_size=(2, 2, 2, 2),
    patch_size=(6, 6, 6, 1),
    depths=(2, 2, 2),
    num_heads=(3, 6, 12),
    downsample="mergingv2",
)

x = torch.randn(2, 2, 96, 96, 96, 20)  # b, c, h, w, d, t
z = model(x)
#print(z.shape)
loss = z.sum()
loss.backward()
print('done')

