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

def log_hyperparameters(args):
    print('\n\n*********Hyperparameters*********\n');
    print(f'batch-size: {args.batch_size}');
    print(f'crop-size: ({args.crop_size_w}, {args.crop_size_h} , {args.crop_size_d})');
    print(f'learning-rate: {args.learning_rate}');
    print(f'sample-per-mri: {args.sample_per_mri}');
    print(f'bl-multiplier: {args.bl_multiplier}');
    print(f'epoch: {args.epoch}');
    print(f'network: {args.network}');
    print(f'resume: {args.resume}');
    print(f'elastic-deform-alpha: {args.elastic_deform_alpha}');
    print(f'cube-texture-mean: {args.cube_texture_mean}');
    print(f'cube-texture-variance: {args.cube_texture_variance}');
    print(f'lesion-size-min: {args.lesion_size_min}');
    print(f'lesion-size-max: {args.lesion_size_max}');

    #find a path for summary writer
    tot_exp = len(os.listdir('exp'));
    path_to_sum_wr = os.path.join('exp', f'Experiment-{tot_exp+1}');
    print(f'\nsave to => {path_to_sum_wr}');
    print('*********\n');
    return path_to_sum_wr;


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
            mri, loc, mask, patient_id = batch[0].to(args.device), batch[1], batch[2].to(args.device), batch[3][0]

            pred = model(mri);      
            pred = torch.sigmoid(pred)>0.5;
            pred = pred * mask;
            dataset.update_prediction(pred, patient_id,loc);
    
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
    parser.add_argument('--miccai16-path', default='miccai-2016', type=str, help='path to miccai-16 dataset');
    parser.add_argument('--sample-per-mri', default=8, type=int, help='number of crops from the given MRI');
    parser.add_argument('--num-workers', default=0, type=int, help='num workers for data loader, should be equal to number of CPU cores');
    parser.add_argument('--dataset-size', default='one', help='if "all" use all the available samples in train and test set, if "one" only used one for each set, it is used for debugging purposes');


    parser.add_argument('--device', default='cuda', type=str, help='device to run models on');
    parser.add_argument('--debug-train-data', default=True, action='store_true', help='debug training data for debugging purposes');
    parser.add_argument('--deterministic', default=False, action='store_true', help='if we want to have same augmentation and same datae, for sanity check');
    parser.add_argument('--bl-multiplier', default=10, type=int, help='boundary loss coefficient');
    parser.add_argument('--epoch', default=500, type=int);
    parser.add_argument('--virtual-batch-size', default=1, type=int, help='use it if batch size does not fit GPU memory');
    parser.add_argument('--network', default='VNet', type=str, help='which model to use');
    parser.add_argument('--resume', action='store_true', default=False, help='whether to resume training or start from beginning');
    parser.add_argument('--resume-path', default='Experiment-6', help='path to checkpoint to resume');
    parser.add_argument('--elastic-deform-alpha', default=500.0, type=float, help='magnitude of elastic deformation');
    parser.add_argument('--cube-texture-mean', default=0.95, type=float, help='cube texture mean value');
    parser.add_argument('--cube-texture-variance', default=0.05, type=float, help='cube texture variance');
    parser.add_argument('--lesion-size-min', default=3, type=float, help='minimum lesion size to include in the augmentation process');
    parser.add_argument('--lesion-size-max', default=12000, type=float, help='maximum lesion size to include in the augmentation process');

    args = parser.parse_args();

    #run only once to cache location of train and test mri
    #cache_dataset_miccai16(args);
    
    train_loader, test_loader, test_dataset = get_loader_miccai16(args);

    if args.network == 'VNet':
        model = VNet().to(args.device);
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate);
    scale = torch.cuda.amp.grad_scaler.GradScaler();
    
    path_to_sum_wr = log_hyperparameters(args);

    local_rank = os.environ['LOCAL_RANK'];

    best_dice = 0;
    start_epoch = 0;
    if args.resume is True:
        ckpt = torch.load(os.path.join('exp', args.resume_path, 'resume.ckpt'), map_location=args.device);
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer']);
        best_dice = ckpt['best_loss'];
        start_epoch = ckpt['epoch'];
        print(f'resuming training from epoch: {start_epoch} with best dice: {best_dice}');

    summary_writer = SummaryWriter(path_to_sum_wr);
    for e in range(start_epoch, args.epoch):
        model.train();
        train_loss = train_step(e, model, train_loader, optimizer, scale, args);
        model.eval();
        valid_dice = valid_step(args, model, test_loader, test_dataset, e);

        summary_writer.add_scalar('train/loss', train_loss, e);
        summary_writer.add_scalar('valid/loss', valid_dice, e);

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scale': scale.state_dict(),
            'best_loss': best_dice,
            'epoch': e+1
        }
        torch.save(ckpt, os.path.join(path_to_sum_wr, 'resume.ckpt'));
        
        save_model = False;
        if best_dice < valid_dice:
            save_model = True;
        
        if save_model:
            print(f'new best model found: {valid_dice}')
            best_dice = valid_dice;
            torch.save({'model': model.state_dict(), 
                        'best_loss': best_dice,
                        'log': path_to_sum_wr}, os.path.join(path_to_sum_wr, 'best_model.ckpt'));

