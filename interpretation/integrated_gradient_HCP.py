#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import os
from tqdm import tqdm
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D
from project.module.pl_classifier_v6_JB import LitClassifier_v6
from project.module.utils.data_module4 import fMRIDataModule4
from project.module.utils.data_module_ABCD import fMRIDataModule_ABCD

from pathlib import Path

# def forward(x):
#     return model.clf(model(x))

save_dir = '/pscratch/sd/j/junbeom/Integrated_gradients_nt5/'
jobid = 2010
for i in Path(f'/global/cfs/cdirs/m4244/junbeom/SwinTransformer4D/output/kjb961013/rs-to-task/RSTOT-{jobid}/').glob('checkpt*'):
    ckpt_path = i
ckpt = torch.load(ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

#ckpt['hyper_parameters']['image_path'] = '/global/cfs/cdirs/m4244/7.cleaned_image_MNI_to_TRs/' # ABCD
ckpt['hyper_parameters']['image_path'] = '/global/cfs/cdirs/m4244/HCP_MNI_to_TRs/' #HCP

ckpt['hyper_parameters']['default_root_dir'] = '/global/cfs/cdirs/m4244'
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['batch_size'] = 1
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1

args = ckpt['hyper_parameters']

model = LitClassifier_v6(**args)
model.cuda(0) if torch.cuda.is_available() else model
model.load_state_dict(ckpt['state_dict'])

integrated_gradients  = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)

kwargs = {
    "nt_samples": 4,
    "nt_samples_batch_size": 4,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 4,
}

data_module = fMRIDataModule4(**args)
data_module.setup()
data_module.prepare_data()
test_loader = data_module.test_dataloader()
model.eval()
for idx, data in enumerate(tqdm(test_loader),0):
    subj_name = data['subject_name'][0]
    dataset_name = ckpt['hyper_parameters']['dataset_name']
    tr = data['TR'].item()
    input_ts = data['fmri_sequence'].float().cuda(0)
    label = data['target'].float().cuda(0)
    
    pred = model.forward(input_ts)
    pred_prob = torch.sigmoid(pred)
    pred_int = (pred_prob>0.5).int().item()
    
    target = data['target']
    target_int = target.int().item()
    
    #only choose corrected samples
    
    if pred_int == target_int:
        if target_int == 0:
            if pred_prob <= 0.25:
                file_dir = os.path.join(save_dir, f'new_{dataset_name}_target0')
                os.makedirs(file_dir,exist_ok=True)
                if tr % 100 == 0:
                    file_path = os.path.join(file_dir, f"{subj_name}_{tr}.pt") 
                    if not os.path.exists(file_path):
                        result = noise_tunnel.attribute(input_ts,baselines=input_ts[0,0,0,0,0,0].item(),target=None,**kwargs)
                        result = result.squeeze().cpu()
                        torch.save(result, file_path)
                        print(f'saving {subj_name}_{tr}.pt')
        elif target_int == 1:
            if pred_prob >= 0.75:
                file_dir = os.path.join(save_dir, f"new_{dataset_name}_target1")
                os.makedirs(file_dir,exist_ok=True)
                if tr % 100 == 0:
                    file_path = os.path.join(file_dir, f"{subj_name}_{tr}.pt") 
                    if not os.path.exists(file_path):
                        result = noise_tunnel.attribute(input_ts,baselines=input_ts[0,0,0,0,0,0].item(),target=None,**kwargs)
                        result = result.squeeze().cpu()
                        torch.save(result, file_path)
                        print(f'saving {subj_name}_{tr}.pt')