from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue

def read_data(filename,load_root,save_root,subj_name,count,queue=None,scaling_method=None, fill_zeroback=False):
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    try:
        # load each nifti file
        data, meta = LoadImage()(path)
    except:
        return None
    
    #change this line according to your file names
    save_dir = os.path.join(save_root,subj_name)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    # change this line according to your dataset
    data = data[:, 14:-7, :, :]
    # width, height, depth, time
    # Inspect the fMRI file first using your visualization tool. 
    # Limit the ranges of width, height, and depth to be under 96. Crop the background, not the brain regions. 
    # Each dimension of fMRI registered to MNI space (2mm) is expected to be around 100.
    # You can do this when you load each volume at the Dataset class, including padding backgrounds to fill dimensions under 96.
   
    background = data==0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0 
    # data_temp[~background].min() is expected to be 0 for scaling_method == 'minmax', and minimum z-value for scaling_method == 'z-norm'
    data_global[~background] = data_temp[~background]

    # save volumes one-by-one in fp16 format.
    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))


def main():
    # change two lines below according to your dataset
    dataset_name = 'ABCD'
    load_root = '/storage/4.cleaned_image' # This folder should have fMRI files in nifti format with subject names. Ex) sub-01.nii.gz 
    save_root = f'/storage/7.{dataset_name}_MNI_to_TRs_minmax'
    scaling_method = 'z-norm' # choose either 'z-norm'(default) or 'minmax'.

    # make result folders
    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root,'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root,'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_root = os.path.join(save_root,'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        subj_name = filename[:-7] 
        # extract subject name from nifti file. [:-7] rules out '.nii.gz'
        # we recommend you use subj_name that aligns with the subject key in a metadata file.

        expected_seq_length = 1000 # Specify the expected sequence length of fMRI for the case your preprocessing stopped unexpectedly and you try to resume the preprocessing.
        
        # change the line below according to your folder structure
        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root,subj_name))) < expected_seq_length): # preprocess if the subject folder does not exist, or the number of pth files is lower than expected sequence length. 
            try:
                count+=1
                p = Process(target=read_data, args=(filename,load_root,save_root,subj_name,count,queue,scaling_method))
                p.start()
                if count % 32 == 0: # requires more than 32 cpu cores for parallel processing
                    p.join()
            except Exception:
                print('encountered problem with'+filename)
                print(Exception)

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
