from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue

def read_abcd(dir,load_root,save_root,count,queue=None,scaling_method=None, fill_zeroback=False):
    print("processing: " + dir, flush=True)
    path = os.path.join(load_root, dir)
    try:
        data, meta = LoadImage()(path)
    except:
        return None
    
    #change this line according to your file names
    save_dir = os.path.join(save_root,dir[14:-7])
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    # change this line according to your dataset
    data = data[1:-2, 14:-7, :, 20:] #It was once [10:-10, 10:-10,0:-10,20:] 
    # it should be 96, 97, 95
    # please crop at the end of y dim by 1, and pad 1 to z dim to make dimension 96 * 96 * 96
    background = data==0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0 
    # data_temp[~background].min() is expected to be ..
    # 0 for scaling_method == 'minmax'
    # minimum z-value for scaling_method == 'z-norm'
    data_global[~background] = data_temp[~background]

    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))


def main():
    # change two lines below according to your dataset
    load_root = '/storage/bigdata/ABCD/fmriprep/1.rs_fmri/4.cleaned_image'
    save_root = '/storage/bigdata/ABCD/fmriprep/1.rs_fmri/7.ABCD_MNI_to_TRs_minmax'
    scaling_method = 'z-norm' #'minmax'

    dirs = os.listdir(load_root)
    os.makedirs(os.path.join(save_root,'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root,'metadata'), exist_ok = True)
    save_root = os.path.join(save_root,'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for dir in sorted(dirs):
        # change the line below according to your folder structure
        if (dir[14:-7] not in finished_samples) or (len(os.listdir(os.path.join(save_root,dir[14:-7]))) < 353):
            try:
                count+=1
                p = Process(target=read_abcd, args=(dir,load_root,save_root,count,queue,scaling_method))
                p.start()
                if count % 32 == 0: # requires more than 32 cpu cores for parallel processing
                    p.join()
            except Exception:
                print('encountered problem with'+dir)
                print(Exception)

if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
