from glob import glob
import os
import nibabel as nib
import torch.amp
from tqdm import tqdm
import argparse
from monai.networks.nets.vnet import VNet
from data_utils import cache_dataset_miccai16, get_loader_miccai16, get_dataset_lesion_size_dist
import torch
from monai.losses.dice import DiceLoss
from utility import BounraryLoss
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
import matplotlib.pyplot as plt
import math



def valid_step(args, model, loader, dataset):
    """one epoch of validating new lesion segmentation model

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    loader : DataLoader
        data loader to iterate through
    
    dataset : Dataset
        used to append patches together and calculate the final results
    """
    print(('\n' + '%10s') %('Dice'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, loc, mask, patient_id = batch[0].to(args.device), batch[1], batch[2].to(args.device), batch[3][0]

            pred = model(mri);      
            pred = torch.sigmoid(pred) > 0.5;
            pred = pred * mask;
            dataset.update_prediction(pred, patient_id,loc);
    
    epoch_dice = dataset.calculate_metrics();
    return epoch_dice;

def visualize_results_arr():
    results_arr = pickle.load(open('results_arr-baseline.dmp', 'rb'));
    results_arr = [[i,a] for i,a in enumerate(results_arr) if math.isnan(a) is False];
    ra = np.arange(3, 342826, 3000)
    ra = [f'({ra[r]}, {ra[r+1]})' for r in range(0, len(ra)-1)];
    indices = np.array(results_arr)[:,0].astype('int');
    res = np.array(results_arr)[:,1]
    ra = [r for idx,r in enumerate(ra) if idx in indices];

    results_arr = [];
    for i in range(len(res)):
       results_arr.append([ra[i],res[i]]); 
    pickle.dump(results_arr, open('results_baseline.dmp', 'wb'))

    plt.bar(ra, res);
    plt.show();

def parse_results_file(file_name):
    with open(file_name, 'r',  encoding="utf8") as f:
        lines = f.readlines();
    ids = [i for i in range(len(lines)) if 'results' in lines[i]]
    results_arr = [];
    ra = pickle.load(open('results_baseline.dmp', 'rb'));
    for i in range(len(ids)):
        results_arr.append([ra[i][0], float(lines[ids[i]][lines[ids[i]].rfind(':')+1:].rstrip())]);
    pickle.dump(results_arr, open(f'results_arr_{file_name}.dmp', 'wb'));



if __name__ == "__main__":

    # file_list = glob('*.out');
    # for f in file_list:
    #     parse_results_file(f);
    parser = argparse.ArgumentParser(description='LesionSegmentation', allow_abbrev=False);
    parser.add_argument('--batch_size', default=4, type=int);
    parser.add_argument('--crop-size-w', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-h', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-d', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--learning-rate', default=1e-4, type=float);
    parser.add_argument('--miccai16-path', default='miccai-2016', type=str, help='path to miccai-16 dataset');
    parser.add_argument('--sample-per-mri', default=8, type=int, help='number of crops from the given MRI');
    parser.add_argument('--num-workers', default=0, type=int, help='num workers for data loader, should be equal to number of CPU cores');
    parser.add_argument('--dataset-size', default='one', help='if "all" use all the available samples in train and test set, if "one" only used one for each set, it is used for debugging purposes');
    parser.add_argument('--device', default='cuda', type=str, help='device to run models on');
    parser.add_argument('--debug-train-data', default=False, action='store_true', help='debug training data for debugging purposes');
    parser.add_argument('--deterministic', default=False, action='store_true', help='if we want to have same augmentation and same datae, for sanity check');
    parser.add_argument('--bl-multiplier', default=10, type=int, help='boundary loss coefficient');
    parser.add_argument('--epoch', default=500, type=int);
    parser.add_argument('--virtual-batch-size', default=1, type=int, help='use it if batch size does not fit GPU memory');
    parser.add_argument('--network', default='VNet', type=str, help='which model to use');
    parser.add_argument('--model-path', default='exp/Net=VNet-baseline-lr1e-3/best_model.ckpt', type=str, help='trained model path');
    parser.add_argument('--lesion-size-min', default=3, type=str, help='trained model path');
    parser.add_argument('--lesion-size-max', default=9999, type=str, help='trained model path');

    args = parser.parse_args();
    
    #get_dataset_lesion_size_dist();
   # visualize_results_arr();

    results_arr = [];
    ra = np.arange(3, 342826, 3000) #342826 is maximum lesion size in the dataset
    for r in range(0, len(ra)-1):
        train_loader, test_loader, test_dataset = get_loader_miccai16(args, train=False, lesion_size_range=(ra[r],ra[r+1]));

        if (test_dataset.total_lesions_size) > 0:
            if args.network == 'VNet':
                model = VNet().to(args.device);
                EXP_NAME = f"Net={args.network}-baseline";
            
            #load model
            ckpt = torch.load(os.path.join(args.model_path, 'best_model.ckpt'), map_location=args.device);
            model.load_state_dict(ckpt['model']);
            
        
            model.eval();
            valid_dice = valid_step(args, model, test_loader, test_dataset);
            print(f'results for ({ra[r], ra[r+1]}): {valid_dice}');
            results_arr.append([f'({ra[r], ra[r+1]})',valid_dice]);

pickle.dump(results_arr, open(f'results_arr_{os.path.basename(args.model_path)}.dmp', 'wb'));


