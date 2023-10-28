cd .. # move to where 'SwiFT is located'
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate py39
 
TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name S1200 --image_path {image_path}' 
DATA_ARGS='--batch_size 8 --num_workers 8  --input_type rest'
DEFAULT_ARGS='--project_name {neptune project name}'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task sex' #--use_scheduler --gamma 0.5 --cycle 0.5' 
RESUME_ARGS=''

export NEPTUNE_API_TOKEN="{neptune API token}" # when using neptune as a logger

export CUDA_VISIBLE_DEVICES={GPU number}

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20

