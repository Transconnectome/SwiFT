<div align="center">    
 
# SwiFT: Swin 4D fMRI Transformer

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

</div>


## 📌&nbsp;&nbsp;Introduction
This project is a collaborative research effort between Seoul National University's M.IN.D Lab (PI: Taesup Moon) and Connectome Lab (PI: Jiook Cha), with the goal of developing a scalable analysis model for fMRI. SwiFT, based on the Swin Transformer, can effectively predict various biological and cognitive variables from fMRI scans and even explain these predictions. We plan to release a large-scale pretrained SwiFT model in the near future, which we hope will assist many researchers using deep learning for fMRI analysis. You can find the research paper at [SwiFT](https://arxiv.org/abs/2307.05916). Feel free to ask any questions regarding this project to the authors. 

**Contact**
- First authors
  - Peter Yongho Kim: peterkim98@snu.ac.kr
  - Junbeom Kwon: kjb961013@snu.ac.kr
- Corresponding authors
  - Professor Taesup Moon: tsmoon@snu.ac.kr
  - Professor Jiook Cha: connectome@snu.ac.kr


> Effective usage of this repository requires learning a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai). Knowledge of some experiment logging frameworks like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) is also recommended.

For someone unfamiliar with these packages, we included an example in the `tutorial_vit/` directory. Briefly, this code trains a ViT model on CIFAR10 with PyTorch lightning modules and logs it into `Tensorboard` or `Neptune.ai`. The core classes are as follows.
- `CIFAR10DataModule`: A class that encapsulates all the steps needed to process data.
- `LitClassifier`: A class that encapsulates the following things: model, train & valid & test steps, and optimizers.
- `pl.Trainer`: A class that contains all of the other processes to operate the `LitClassifier`.
You can easily run the following code to train a ViT model directly.
```bash
bash scripts/tutorial.sh
 ```  
---

## 1. Description
This repository implements the Swin 4D fMRI transformer (SwiFT). 
- Our code offers the following things.
  - Trainer based on PyTorch Lightning for running SwiFT.
  - `SwinTransformer4D` architecture and its variants
  - Data preprocessing/loading pipelines for 4D fMRI datasets.
  - Self-supervised learning strategies


## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/Transconnectome/SwiFT.git

# install project   
cd SwiFT
conda env create -f envs/py39.yaml
conda activate py39
 ```

## 3. Project Structure
Our directory structure looks like this:

```
├── notebooks                    <- Useful Jupyter notebook examples are given (TBU)
├── output                       <- Experiment log and checkpoints will be saved here once you train a model
├── envs                         <- Conda environment
├── pretrained_models            <- Pretrained model checkpoints
│   ├── contrastive_pretrained.ckpt        <- Contrastively pretrained model on all three datasets 
│   ├── hcp_sex_classification.ckpt        <- Model trained for the sex classification task on HCP dataset 
│   ├── split_hcp.txt                      <- Data split for the trained HCP model 
├── project                 
│   ├── module                   <- Every module is given in this directory
│   │   ├── models               <- Models (Swin fMRI Transformer)
│   │   ├── utils                
│   │   │    ├── data_module.py  <- Dataloader & codes for matching fMRI scans and target variables
│   │   │    └── data_preprocessing_and_load
│   │   │        ├── datasets.py           <- Dataset Class for each dataset
│   │   │        └── preprocessing.py      <- Preprocessing codes for step 6
│   │   └── pl_classifier.py    <- LightningModule
│   └── main.py                 <- Main code that trains and tests the 4DSwinTransformer model
│
├── test                 
│   ├── module_test_swin.py     <- Code for debugging SwinTransformer
│   └── module_test_swin4d.py   <- Code for debugging 4DSwinTransformer
│ 
├── sample_scripts              <- Example shell scripts for training
│
├── .gitignore                  <- List of files/folders ignored by git
├── export_DDP_vars.sh          <- setup file for running torch DistributedDataParallel (DDP) 
└── README.md
```

<br>

## 4. Train model

### 4.0 Quick start

- Single forward & backward pass for debugging SwinTransformer4D model.

```bash
cd SwiFT/
python test/module_test_swin4d.py
 ```  

### 4.1 Arguments for trainer
You can check the arguments list by using -h
 ```bash
python project/main.py --data_module dummy --classifier_module default -h
```

```
usage: main.py [-h] [--seed SEED] [--dataset_name {S1200,ABCD,UKB,Dummy}]
               [--downstream_task DOWNSTREAM_TASK]
               [--downstream_task_type DOWNSTREAM_TASK_TYPE]
               [--classifier_module CLASSIFIER_MODULE]
               [--loggername LOGGERNAME] [--project_name PROJECT_NAME]
               [--resume_ckpt_path RESUME_CKPT_PATH]
               [--load_model_path LOAD_MODEL_PATH] [--test_only]
               [--test_ckpt_path TEST_CKPT_PATH] [--freeze_feature_extractor]
               [--grad_clip] [--optimizer OPTIMIZER] [--use_scheduler]
               [--weight_decay WEIGHT_DECAY] [--learning_rate LEARNING_RATE]
               [--momentum MOMENTUM] [--gamma GAMMA] [--cycle CYCLE]
               [--milestones MILESTONES [MILESTONES ...]] [--adjust_thresh]
               [--use_contrastive] [--contrastive_type CONTRASTIVE_TYPE]
               [--pretraining] [--augment_during_training]
               [--augment_only_affine] [--augment_only_intensity]
               [--temperature TEMPERATURE] [--model MODEL]
               [--in_chans IN_CHANS] [--embed_dim EMBED_DIM]
               [--window_size WINDOW_SIZE [WINDOW_SIZE ...]]
               [--first_window_size FIRST_WINDOW_SIZE [FIRST_WINDOW_SIZE ...]]
               [--patch_size PATCH_SIZE [PATCH_SIZE ...]]
               [--depths DEPTHS [DEPTHS ...]]
               [--num_heads NUM_HEADS [NUM_HEADS ...]]
               [--c_multiplier C_MULTIPLIER]
               [--last_layer_full_MSA LAST_LAYER_FULL_MSA]
               [--clf_head_version CLF_HEAD_VERSION]
               [--attn_drop_rate ATTN_DROP_RATE] [--scalability_check]
               [--process_code PROCESS_CODE]
               [--dataset_split_num DATASET_SPLIT_NUM]
               [--label_scaling_method {minmax,standardization}]
               [--image_path IMAGE_PATH] [--bad_subj_path BAD_SUBJ_PATH]
               [--input_type {rest,task}] [--train_split TRAIN_SPLIT]
               [--val_split VAL_SPLIT] [--batch_size BATCH_SIZE]
               [--eval_batch_size EVAL_BATCH_SIZE]
               [--img_size IMG_SIZE [IMG_SIZE ...]]
               [--sequence_length SEQUENCE_LENGTH]
               [--stride_between_seq STRIDE_BETWEEN_SEQ]
               [--stride_within_seq STRIDE_WITHIN_SEQ]
               [--num_workers NUM_WORKERS] [--with_voxel_norm WITH_VOXEL_NORM]
               [--shuffle_time_sequence]
               [--limit_training_samples LIMIT_TRAINING_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seeds. recommend aligning this argument with
                        data split number to control randomness (default:
                        1234)
  --dataset_name {S1200,ABCD,UKB,Dummy}
  --downstream_task DOWNSTREAM_TASK
                        downstream task (default: sex)
  --downstream_task_type DOWNSTREAM_TASK_TYPE
                        select either classification or regression according
                        to your downstream task (default: default)
  --classifier_module CLASSIFIER_MODULE
                        A name of lightning classifier module (outdated
                        argument) (default: default)
  --loggername LOGGERNAME
                        A name of logger (default: default)
  --project_name PROJECT_NAME
                        A name of project (Neptune) (default: default)
  --resume_ckpt_path RESUME_CKPT_PATH
                        A path to previous checkpoint. Use when you want to
                        continue the training from the previous checkpoints
                        (default: None)
  --load_model_path LOAD_MODEL_PATH
                        A path to the pre-trained model weight file (.pth)
                        (default: None)
  --test_only           specify when you want to test the checkpoints (model
                        weights) (default: False)
  --test_ckpt_path TEST_CKPT_PATH
                        A path to the previous checkpoint that intends to
                        evaluate (--test_only should be True) (default: None)
  --freeze_feature_extractor
                        Whether to freeze the feature extractor (for
                        evaluating the pre-trained weight) (default: False)

Default classifier:
  --grad_clip           whether to use gradient clipping (default: False)
  --optimizer OPTIMIZER
                        which optimizer to use [AdamW, SGD] (default: AdamW)
  --use_scheduler       whether to use scheduler (default: False)
  --weight_decay WEIGHT_DECAY
                        weight decay for optimizer (default: 0.01)
  --learning_rate LEARNING_RATE
                        learning rate for optimizer (default: 0.001)
  --momentum MOMENTUM   momentum for SGD (default: 0)
  --gamma GAMMA         decay for exponential LR scheduler (default: 1.0)
  --cycle CYCLE         cycle size for CosineAnnealingWarmUpRestarts (default:
                        0.3)
  --milestones MILESTONES [MILESTONES ...]
                        lr scheduler (default: [100, 150])
  --adjust_thresh       whether to adjust threshold for valid/test (default:
                        False)
  --use_contrastive     whether to use contrastive learning (specify
                        --contrastive_type argument as well) (default: False)
  --contrastive_type CONTRASTIVE_TYPE
                        combination of contrastive losses to use [1: Use the
                        Instance contrastive loss function, 2: Use the local-
                        local temporal contrastive loss function, 3: Use the
                        sum of both loss functions] (default: 0)
  --pretraining         whether to use pretraining (default: False)
  --augment_during_training
                        whether to augment input images during training
                        (default: False)
  --augment_only_affine
                        whether to only apply affine augmentation (default:
                        False)
  --augment_only_intensity
                        whether to only apply intensity augmentation (default:
                        False)
  --temperature TEMPERATURE
                        temperature for NTXentLoss (default: 0.1)
  --model MODEL         which model to be used (default: none)
  --in_chans IN_CHANS   Channel size of input image (default: 1)
  --embed_dim EMBED_DIM
                        embedding size (recommend to use 24, 36, 48) (default:
                        24)
  --window_size WINDOW_SIZE [WINDOW_SIZE ...]
                        window size from the second layers (default: [4, 4, 4,
                        4])
  --first_window_size FIRST_WINDOW_SIZE [FIRST_WINDOW_SIZE ...]
                        first window size (default: [2, 2, 2, 2])
  --patch_size PATCH_SIZE [PATCH_SIZE ...]
                        patch size (default: [6, 6, 6, 1])
  --depths DEPTHS [DEPTHS ...]
                        depth of layers in each stage (default: [2, 2, 6, 2])
  --num_heads NUM_HEADS [NUM_HEADS ...]
                        The number of heads for each attention layer (default:
                        [3, 6, 12, 24])
  --c_multiplier C_MULTIPLIER
                        channel multiplier for Swin Transformer architecture
                        (default: 2)
  --last_layer_full_MSA LAST_LAYER_FULL_MSA
                        whether to use full-scale multi-head self-attention at
                        the last layers (default: False)
  --clf_head_version CLF_HEAD_VERSION
                        clf head version, v2 has a hidden layer (default: v1)
  --attn_drop_rate ATTN_DROP_RATE
                        dropout rate of attention layers (default: 0)
  --scalability_check   whether to check scalability (default: False)
  --process_code PROCESS_CODE
                        Slurm code/PBS code. Use this argument if you want to
                        save process codes to your log (default: None)

DataModule arguments:
  --dataset_split_num DATASET_SPLIT_NUM
  --label_scaling_method {minmax,standardization}
                        label normalization strategy for a regression task
                        (mean and std are automatically calculated using train
                        set) (default: standardization)
  --image_path IMAGE_PATH
                        path to image datasets preprocessed for SwiFT
                        (default: None)
  --bad_subj_path BAD_SUBJ_PATH
                        path to txt file that contains subjects with bad fMRI
                        quality (default: None)
  --input_type {rest,task}
                        refer to datasets.py (default: rest)
  --train_split TRAIN_SPLIT
  --val_split VAL_SPLIT
  --batch_size BATCH_SIZE
  --eval_batch_size EVAL_BATCH_SIZE
  --img_size IMG_SIZE [IMG_SIZE ...]
                        image size (adjust the fourth dimension according to
                        your --sequence_length argument) (default: [96, 96,
                        96, 20])
  --sequence_length SEQUENCE_LENGTH
  --stride_between_seq STRIDE_BETWEEN_SEQ
                        skip some fMRI volumes between fMRI sub-sequences
                        (default: 1)
  --stride_within_seq STRIDE_WITHIN_SEQ
                        skip some fMRI volumes within fMRI sub-sequences
                        (default: 1)
  --num_workers NUM_WORKERS
  --with_voxel_norm WITH_VOXEL_NORM
  --shuffle_time_sequence
  --limit_training_samples LIMIT_TRAINING_SAMPLES
                        use if you want to limit training samples (default:None)
```

### 4.2 Hidden Arguments for PyTorch lightning
pytorch_lightning offers useful arguments for training. For example, we used `--max_epochs` and `--default_root_dir` in our experiments. We recommend the user refer to the following link to check the argument lists.

([https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer](https://lightning.ai/docs/pytorch/1.8.6/common/trainer.html))

### 4.3 Commands/scripts for running classification/regression tasks
- Training SwiFT in an interactive way
  
 ```bash
# interactive
cd SwiFT/
bash sample_scripts/sample_script.sh
```
This bash script was tested on the server cluster (Linux) with 8 RTX 3090 GPUs.
You should correct the following lines.

 ```bash
cd {path to your 'SwiFT' directory}
source /usr/anaconda3/etc/profile.d/conda.sh (init conda) # might change if you have your own conda.
conda activate {conda env name}
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name {dataset_name} --image_path {path to the image data}' # This script assumes that you have preprocessed HCP dataset. You may run the codes anyway with "--dataset_name Dummy"
DEFAULT_ARGS='--project_name {neptune project name}'
export NEPTUNE_API_TOKEN="{Neptune API token allocated to each user}"
export CUDA_VISIBLE_DEVICES={usable GPU number}
 ```

- Training SwiFT with Slurm (if you run the codes at Slurm-based clusters)
Please refer to the [tutorial](https://slurm.schedmd.com/sbatch.html) for Slurm commands.
 ```bash
cd SwiFT/
sbatch sample_scripts/sample_script.slurm
```

### 4.4 Commands for the self-supervised pertaining
To perform self-supervised pre-training, add the following arguments to the base script (you can change the contrastive type):
```bash
--pretraining --use_contrastive --contrastive_type 1
```

## 5. Loggers
We offer two options for loggers.
- Tensorboard (https://www.tensorflow.org/tensorboard)
   - Log & model checkpoints are saved in `--default_root_dir`
   - Logging test code with Tensorboard is not available.
- Neptune AI (https://neptune.ai/)
   - Generate a new workspace and project on the Neptune website.
      - Academic workspace offers 200GB of storage and collaboration for free. 
   - export NEPTUNE_API_TOKEN="YOUR API TOKEN" in your script.
   - specify the "--project_name" argument with your Neptune project name. ex) "--project_name user-id/project"


## 6. How to prepare your own dataset
These preprocessing codes are implemented based on the initial repository by GonyRosenman [TFF](https://github.com/GonyRosenman/TFF)

To make your own dataset, you should execute either of the minimal preprocessing steps:
- fMRIprep [Preprocessing with fMRIprep](https://fmriprep.org/en/stable/)
- FSL [UKB Preprocessing pipeline](https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf)

 * We ensure that each brain is registered to the MNI space, and the whole brain mask is applied to remove non-brain regions. 
 * We are investigating how additional preprocessing steps to remove confounding factors such as head movement impact performance.

After the minimal preprocessing steps, you should perform additional preprocessing to use SwiFT. (You can find the preprocessing code at 'project/module/utils/data_preprocessing_and_load/preprocessing.py')
- normalization: voxel normalization(not used) and whole-brain z-normalization (mainly used)
- change fMRI volumes to floating point 16 to save storage and decrease IO bottleneck.
- each fMRI volume is saved separately as torch checkpoints to facilitate window-based training.
- remove non-brain(background) voxels that are over 96 voxels.
   - you should open your fMRI scans to determine the level that does not cut out the brain regions
   - you can use `nilearn` to visualize your fMRI data. (official documentation: [here](https://nilearn.github.io/dev/index.html))
  ```python
  from nilearn import plotting
  from nilearn.image import mean_img
  
  plotting.view_img(mean_img(fmri_filename), threshold=None)
  ```
   - if your dimension is under 96, you can pad non-brain voxels at 'datasets.py' files.

* refer to the annotation in the 'preprocessing.py' code to adjust it for your own datasets.

The resulting data structure is as follows:
```
├── {Dataset name}_MNI_to_TRs                 
   ├── img                  <- Every normalized volume is located in this directory
   │   ├── sub-01           <- subject name
   │   │  ├── frame_0.pt    <- Each torch pt file contains one volume in a fMRI sequence (total number of pt files = length of fMRI sequence)
   │   │  ├── frame_1.pt
   │   │  │       :
   │   │  └── frame_{T}.pt  <- the last volume in an fMRI sequence (length T) 
   │   └── sub-02              
   │   │  ├── frame_0.pt    
   │   │  ├── frame_1.pt
   │   │  ├──     :
   └── metadata
       └── metafile.csv     <- file containing target variable
```

## 7. Define the Dataset class for your own dataset.
* The data loading pipeline works by processing image and metadata at 'project/module/utils/data_module.py' and passing the paired image-label tuples to the Dataset classes at 'project/module/utils/data_preprocessing_and_load/datasets.py.'
* you should implement codes for combining image path, subject_name, and target variables at 'project/module/utils/data_module.py'
* you should define Dataset Class for your dataset at 'project/module/utils/data_preprocessing_and_load/datasets.py.' In the Dataset class (__getitem__), you should specify how many background voxels you would add or remove to make the volumes shaped 96 * 96 * 96.

## 8. Pretrained model checkpoints
We provide some pretrained model checkpoints under the pretrained_models directory.
* contrastive_pretrained.ckpt contains a contrastively pre-trained model using all three datasets: HCP, ABCD, and UKB.
* hcp_sex_classification.ckpt contains the model trained from scratch for the sex classification task on the HCP dataset.
* split_hcp.txt contains the train, validation, test split used for training the hcp_sex_classification.ckpt model.
* To fine-tune the provided models for another task, use the load_model_path argument on main.py

Please contact the authors if you have any additional requests for the pretrained model checkpoints.

## 9. Scalability testing
We allow some functionalities that can check the scalability of SwiFT.

* For using Dummy Dataset, you should specify '--dataset_name' argument as 'Dummy'
* For checking the effect of the training sample, you can limit the number of subjects for model training by specifying the number for the '--limit_training_samples' argument.
* For checking the throughput following the number of GPUs/nodes, you should add the '--scalability_check' argument to your script.


### Citation   
```
@article{kim2023swift,
  title={SwiFT: Swin 4D fMRI Transformer},
  author={Kim, Peter Yongho and Kwon, Junbeom and Joo, Sunghwan and Bae, Sangyoon and Lee, Donggyu and Jung, Yoonho and Yoo, Shinjae and Cha, Jiook and Moon, Taesup},
  journal={arXiv preprint arXiv:2307.05916},
  year={2023}
}
```   
