from glob import glob
import os
import nibabel as nib
import torch.amp
from tqdm import tqdm
import argparse
from monai.networks.nets.vnet import VNet
from data_utils import cache_dataset_miccai16, get_loader_miccai16
import torch
from monai.losses.dice import DiceLoss
from utility import BounraryLoss
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def cache_miccai16_dataset():
    miccai_16_root = "../SSLMRI/miccai-2016";
    l1_folders = glob(miccai_16_root+'/*/');
    for l1f in l1_folders:
        l2_folders = glob(os.path.join(l1f , '*/'));
        for l2f in l2_folders:
            l3_folders = glob(os.path.join(l2f,  '*/'));
            for l3f in tqdm(l3_folders):
                if os.path.exists(os.path.join(l3f, 'Preprocessed_Data', 'T1_preprocessed_seg.nii.gz')) is False:
                    t1 = ((os.path.join(l3f, 'Preprocessed_Data', 'T1_preprocessed.nii.gz')));
                    output_path = os.path.join(l3f, 'Preprocessed_Data', 'T1_preprocessed_segmentation.nii.gz')
                    os.system(f"fast  -n 3 -t 1 {t1}");

def cache_isbi_dataset():
    isbi_root = "../SSLMRI/isbi";
    l1_folders = glob(isbi_root+'/*/')[1:];
    for i, l1f in enumerate(l1_folders):
        num_files = len(os.listdir(os.path.join(l1f, 'preprocessed')));
        for l2f in tqdm(range(1, int(num_files/4)+1)):
            if os.path.exists(os.path.join(l1f, 'preprocessed', f'training0{i+2}_0{l2f}_t2_pp_seg.nii.gz')) is False:
                t1 = ((os.path.join(l1f, 'preprocessed', f'training0{i+2}_0{l2f}_t2_pp.nii.gz')));
                os.system(f"fast  -n 3 -t 2 {t1}");

def train_step(epoch, model, loader, optimizer, scaler, args):
    print(('\n' + '%10s' *3)%('Epoch', 'Loss', 'IoU'))
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}' );
    epoch_loss = [];
    epoch_IoU = [];
    curr_step = 0;
    curr_iou = 0;

    for idx, batch in pbar:
        mri, mask, dt = batch[0].squeeze(0).to(args.device), batch[1].squeeze(0).to(args.device), batch[2].squeeze(0).to(args.device);
        steps = args.sample_per_mri // args.batch_size;
        for s in range(steps):
            curr_mri = mri[s*args.batch_size:(s+1)*args.batch_size]
            curr_mask = mask[s*args.batch_size:(s+1)*args.batch_size]
            curr_dt = dt[s*args.batch_size:(s+1)*args.batch_size]
            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(curr_mri);
                dl = DiceLoss(sigmoid=True)(pred, curr_mask);
                bl = BounraryLoss(sigmoid=True)(pred, curr_dt)*args.bl_multiplier;
                loss = dl + bl;

            scaler.scale(loss).backward();
            curr_loss = loss.item();
            curr_step+=1;
            curr_iou += (1-(DiceLoss(sigmoid=True)(pred, curr_mask)).item());

            if (curr_step) % args.virtual_batch_size == 0:
                scaler.step(optimizer);
                scaler.update();
                
                model.zero_grad(set_to_none = True);
                epoch_loss.append(curr_loss);
                epoch_IoU.append(curr_iou);
                curr_loss = 0;
                curr_step = 0;
                curr_iou = 0;

            pbar.set_description(('%10s' + '%10.4g'*2)%(epoch, np.mean(epoch_loss), np.mean(epoch_IoU)));

    return np.mean(epoch_loss);


def valid_step(args, model, loader, dataset, epoch):
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
    print(('\n' + '%10s'*2) %('Epoch', 'Dice'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, loc = batch[0].to(args.device), batch[1]

            pred = model(mri);      
            pred = torch.sigmoid(pred)>0.5;
            dataset.update_prediction(pred, loc);
    
    epoch_dice = dataset.calculate_metrics();
    print(('\n' + '%10i' + '%10f') %(epoch, epoch_dice));
    return epoch_dice;

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LesionSegmentation', allow_abbrev=False);
    parser.add_argument('--batch_size', default=4, type=int);
    parser.add_argument('--crop-size-w', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-h', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--crop-size-d', default=96, type=int, help='crop size for getting a patch from MRI scan');
    parser.add_argument('--learning-rate', default=1e-4, type=float);
    parser.add_argument('--miccai16-path', default='C:/PhD/Thesis/MRIProject/SSLMRI/miccai-2016', type=str, help='path to miccai-16 dataset');
    parser.add_argument('--sample-per-mri', default=8, type=int, help='number of crops from the given MRI');
    parser.add_argument('--num-workers', default=0, type=int, help='num workers for data loader, should be equal to number of CPU cores');
    parser.add_argument('--use-one-sample-only', default=True, action='store_true');
    parser.add_argument('--device', default='cuda', type=str, help='device to run models on');
    parser.add_argument('--debug-train-data', default=False, action='store_true', help='debug training data for debugging purposes');
    parser.add_argument('--deterministic', default=False, action='store_true', help='if we want to have same augmentation and same datae, for sanity check');
    parser.add_argument('--bl-multiplier', default=10, type=int, help='boundary loss coefficient');
    parser.add_argument('--epoch', default=500, type=int);
    parser.add_argument('--virtual-batch-size', default=1, type=int, help='use it if batch size does not fit GPU memory');
    parser.add_argument('--network', default='VNet', type=str, help='which model to use');



    args = parser.parse_args();

    #run only once to cache location of train and test mri
    cache_dataset_miccai16(args);
    

    

    train_loader, test_loader, test_dataset = get_loader_miccai16(args);

    if args.network == 'VNet':
        model = VNet().to(args.device);
        EXP_NAME = f"Net={args.network}-baseline";
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate);
    scale = torch.cuda.amp.grad_scaler.GradScaler();
    
    print(EXP_NAME);
    summary_writer = SummaryWriter(os.path.join('exp', EXP_NAME));
    best_dice = 0;
    for e in range(args.epoch):
        model.train();
        train_loss = train_step(e, model, train_loader, optimizer, scale, args);
        model.eval();
        valid_dice = valid_step(args, model, test_loader, test_dataset, e);

        summary_writer.add_scalar('train/loss', train_loss, e);
        summary_writer.add_scalar('valid/loss', valid_dice, e);

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_dice,
            'epoch': e+1
        }
        torch.save(ckpt,os.path.join('exp', EXP_NAME, 'resume.ckpt'));
        
        save_model = False;
        if best_dice < valid_dice:
            save_model = True;
        
        if save_model:
            print(f'new best model found: {valid_dice}')
            best_dice = valid_dice;
            torch.save({'model': model.state_dict(), 
                        'best_loss': best_dice,
                        'log': EXP_NAME}, os.path.join('exp', EXP_NAME, 'best_model.ckpt'));

