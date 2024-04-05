import os
from torch.utils.data import DataLoader, Dataset
import pickle
from glob import glob
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import nibabel as nib
from skimage.filters import sobel, threshold_otsu
from monai.transforms import Compose, Resize, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise, RandSpatialCropSamplesd
from scipy.ndimage import binary_opening
from tqdm import tqdm
import math
from patchify import patchify
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter
from utility import remove_small_regions, calculate_metric_percase

def window_center_adjustment(img):
    """window center adjustment, similar to what ITKSnap does

    Parameters
    ----------
    img : np.ndarray
        input image

    """
    hist = np.histogram(img.ravel(), bins = int(np.max(img)))[0];
    hist = hist / (hist.sum()+1e-4);
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;

def cache_dataset_miccai16(args):
    """cache testing dataset for self-supervised pretraining model

    Parameters
    ----------
    args : dict
        arguments

    """
    if os.path.exists(f'cache_miccai-2016') is False:
        os.makedirs(f'cache_miccai-2016');
    
    training_centers = ['01', '07', '08'];
    testing_centers = ['01', '03', '07', '08'];
    train_mri = [];
    test_mri = [];
    all_mri_path = [];
    for tc in training_centers:
        patients = glob(os.path.join(args.miccai16_path, 'Training', f'Center_{tc}','*/'));
        for p in patients:
            train_mri.append(os.path.join(p, 'Preprocessed_Data', 'FLAIR_preprocessed.nii.gz').replace('\\', '/'));
    
    for tc in testing_centers:
        patients = glob(os.path.join(args.miccai16_path, 'Testing', f'Center_{tc}','*/'));
        for p in patients:
            test_mri.append(os.path.join(p, 'Preprocessed_Data', 'FLAIR_preprocessed.nii.gz').replace('\\', '/'));



    pickle.dump([train_mri, test_mri], open(os.path.join('cache_miccai-2016', 'train_test_split.dmp'), 'wb'));

def cropper(mri, 
            mask,
            gt, 
            roi_size,
            num_samples):
    """crop two time-points MRI scans at the same time for new lesion segmentation model

    Parameters
    ----------
    mri1 : np.ndarray
        first MRI scan

    mri2 : np.ndarray
        second MRI scan

    gt : np.ndarray
        ground truth segmentations

    gr : np.ndarray
        gradients of MRI scan

    rot_size : list
        size to crop for three axis

    num_samples : int
        number of samples to crop

    """
    ret = [];
    for i in range(num_samples):
        pos_cords = np.where(gt > 0);
        r = np.random.randint(0,len(pos_cords[0]));
        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        d_x_l = min(roi_size[0]//2,center[0]);
        d_x_r = min(roi_size[0]//2 ,mri.shape[1]-center[0]);
        if d_x_l != roi_size[0]//2:
            diff = abs(roi_size[0]//2 - center[0]);
            d_x_r += diff;
        if d_x_r != roi_size[0]//2 and d_x_l == roi_size[0]//2:
            diff = abs(roi_size[0]//2 - (mri.shape[1]-center[0]));
            d_x_l += diff;
        
        d_y_l = min(roi_size[1]//2,center[1]);
        d_y_r = min(roi_size[1]//2 ,mri.shape[2]-center[1]);
        if d_y_l != roi_size[1]//2:
            diff = abs(roi_size[1]//2 - center[1]);
            d_y_r += diff;
        if d_y_r != roi_size[1]//2 and d_y_l == roi_size[1]//2:
            diff = abs(roi_size[1]//2 - mri.shape[2]-center[1]);
            d_y_l += diff;
        
        d_z_l = min(roi_size[2]//2,center[2]);
        d_z_r = min(roi_size[2]//2 ,mri.shape[3]-center[2]);
        if d_z_l != roi_size[2]//2:
            diff = abs(roi_size[2]//2 - center[2]);
            d_z_r += diff;
        if d_z_r != roi_size[2]//2 and d_z_l == roi_size[2]//2:
            diff = abs(roi_size[2]//2 - mri.shape[3]-center[2]);
            d_z_l += diff;

        sign_x = np.random.randint(1,3);
        if sign_x%2!=0:
            offset_x = np.random.randint(0, max(min(abs(center[0]-int(d_x_l)), int(d_x_l//2)),1))*-1;
        else:
            offset_x = np.random.randint(0, max(min(abs(center[0]+int(d_x_r)-mri.shape[1]), int(d_x_r//2)), 1));
        start_x = center[0]-int(d_x_l)+offset_x;
        end_x = center[0]+int(d_x_r)+offset_x;

        sign_y = np.random.randint(1,3);
        if sign_y%2!=0:
            offset_y = np.random.randint(0, max(min(abs(center[1]-int(d_y_l)), int(d_y_l//2)),1))*-1;
        else:
            offset_y = np.random.randint(0, max(min(abs(center[1]+int(d_y_r)-mri.shape[2]), int(d_y_r//2)), 1));
        start_y = center[1]-int(d_y_l) + offset_y;
        end_y = center[1]+int(d_y_r) + offset_y;

        sign_z = np.random.randint(1,3);
        if sign_z%2!=0:
            offset_z = np.random.randint(0, max(min(abs(center[2]-int(d_z_l)), int(d_z_l)),1))*-1;
        else:
            offset_z = np.random.randint(0, max(min(abs(center[2]+int(d_z_r)-mri.shape[3]), int(d_z_r//2)), 1));
        
        start_z = center[2]-int(d_z_l)+offset_z;
        end_z = center[2]+int(d_z_r)+offset_z;

        d = dict();
        temp = mri[:, start_x:end_x, start_y:end_y, start_z:end_z];
        assert temp.shape[1] == 96 and temp.shape[2] == 96 and temp.shape[3] == 96
        d['mri'] = torch.from_numpy(mri[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['mask'] = torch.from_numpy(mask[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['gt'] = torch.from_numpy(gt[:, start_x:end_x, start_y:end_y, start_z:end_z]);

        ret.append(d);

    return ret;

class MICCAI_Dataset(Dataset):
    """Dataset for self-supervised pretraining

    it returns examples for training which includes to MRI patch and one ground truth labels

    Parameters
    ----------
    args : dict
        arguments

    patient_ids : list
        list of mri images, should a string list

    train : bool
        indicate if we are in training or testing mode

    Attributes
    ----------
    mr_imges : list
        list of loaded mri scans

    """
    def __init__(self, 
                 args, 
                 patient_ids, 
                 train = True) -> None:
        super().__init__();
        self.args = args;
        m1 = 0.4;
        m2 = 0.5;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=.05),
            RandGibbsNoise(prob=1.0, alpha=(0.35,0.45))
        ], weights=[1,1,1])


        self.transforms = Compose(
            [
                
                NormalizeIntensity(subtrahend=0.5, divisor=0.5)
            ]
        )

        self.crop_rand = Compose(
            [ 
                RandCropByPosNegLabeld(
                keys=['mri', 'mask', 'gt'], 
                label_key='gt', 
                spatial_size= (args.crop_size_w, args.crop_size_h, args.crop_size_d),
                pos=1, 
                neg=0,
                num_samples=args.sample_per_mri if train else 1,)
            ]
        )

        self.train = train;

        self.data = [];
        
        
        if train:
            for patient_path in patient_ids:
                base_path = patient_path[:patient_path.rfind('/')];
                gt_path = base_path[:base_path.rfind('/')];
                mri = nib.load(os.path.join(patient_path));
                mri = mri.get_fdata();
                mri = window_center_adjustment(mri);

                mask = nib.load(os.path.join(base_path, f'T1_preprocessed_pve_2.nii.gz'));
                mask = mask.get_fdata();

                gt = nib.load(os.path.join(gt_path, 'Masks', f'Consensus.nii.gz'));
                gt = gt.get_fdata();
                
                self.data.append([mri, mask, gt, patient_path]);
        
        else:
            self.pred_data = dict();
            self.gt_data = dict();
            for patient_path in patient_ids:
                base_path = patient_path[:patient_path.rfind('/')];
                gt_path = base_path[:base_path.rfind('/')];
                mri = nib.load(os.path.join(patient_path));
                mri = mri.get_fdata();
                mri = window_center_adjustment(mri);

                mask = nib.load(os.path.join(base_path, f'T1_preprocessed_pve_2.nii.gz'));
                mask = mask.get_fdata();

                gt = nib.load(os.path.join(gt_path, 'Masks', f'Consensus.nii.gz'));
                gt = gt.get_fdata();

                mri = mri / (np.max(mri)+1e-4);


                w,h,d = mri.shape;
                new_w = math.ceil(w / args.crop_size_w) * args.crop_size_w;
                new_h = math.ceil(h / args.crop_size_h) * args.crop_size_h;
                new_d = math.ceil(d / args.crop_size_d) * args.crop_size_d;

                mri_padded  = np.zeros((new_w, new_h, new_d), dtype = mri.dtype);
                mask_padded  = np.zeros((new_w, new_h, new_d), dtype = mask.dtype);
                gt_padded  = np.zeros((new_w, new_h, new_d), dtype = gt.dtype);

                mri_padded[:w,:h,:d] = mri;
                mask_padded[:w,:h,:d] = mask;
                gt_padded[:w,:h,:d] = gt;

                self.step_w, self.step_h, self.step_d = args.crop_size_w, args.crop_size_h, args.crop_size_d;
                mri_patches = patchify(mri_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                mask_patches = patchify(mask_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                gt_patches = patchify(gt_padded, 
                                                    (args.crop_size_w, args.crop_size_h, args.crop_size_d), 
                                                    (self.step_w, self.step_h, self.step_d));
                curr_data = [];
                for i in range(mri_patches.shape[0]):
                    for j in range(mri_patches.shape[1]):
                        for k in range(mri_patches.shape[2]):
                            curr_data.append((mri_patches[i,j,k,...], gt_patches[i,j,k,...], [i,j,k], patient_path))

                predicted_aggregated = np.zeros((new_w, new_h, new_d), dtype = np.int32);
                self.pred_data[patient_path] = predicted_aggregated;
                self.gt_data[patient_path] = gt_padded;

                self.data.extend(curr_data);

            
    
    def __len__(self):
        return len(self.data);

    def __getitem__(self, index):
        if self.train:
            
            mri, mask, gt, path = self.data[index];
            mask = mask > 0;
            mri = np.expand_dims(mri, axis=0);
            mask = np.expand_dims(mask, axis=0);
            #gt = remove_small_regions(gt.squeeze(), min_size=100000);

            gt = np.expand_dims(gt, axis=0);
       
            mri = mri / (np.max(mri)+1e-4);            

            if self.args.deterministic is False:
                # if np.sum(gt) != 0:
                #     ret_transforms = cropper(mri,
                #                              mask, 
                #                              gt,
                #                              roi_size=(self.args.crop_size_w, self.args.crop_size_h, self.args.crop_size_d),
                #                              num_samples=self.args.sample_per_mri if self.train else 1);
                # else:
                ret_transforms = self.crop_rand({'mri': mri, 'mask': mask,'gt': gt});
            
            ret_mri = None;
            ret_gt = None;
            ret_dt = None;

            for i in range(self.args.sample_per_mri):
                if self.args.deterministic is False:
                    mri_c = ret_transforms[i]['mri'];
                    mask_c = ret_transforms[i]['mask'];
                    gt_c = ret_transforms[i]['gt'];

                else:
                    center1 = [int(mri.shape[1]//2),int(mri.shape[2]//2), int(mri.shape[3]//2)]
                    mri_c = torch.from_numpy(mri[:, int(center1[0]-self.args.crop_size_w//2):int(center1[0]+self.args.crop_size_w//2), int(center1[1]-self.args.crop_size_h//2):int(center1[1]+self.args.crop_size_h//2), int(center1[2]-self.args.crop_size_d//2):int(center1[2]+self.args.crop_size_d//2)]);
                    gt_c = torch.from_numpy(gt[:, int(center1[0]-self.args.crop_size_w//2):int(center1[0]+self.args.crop_size_w//2), int(center1[1]-self.args.crop_size_h//2):int(center1[1]+self.args.crop_size_h//2), int(center1[2]-self.args.crop_size_d//2):int(center1[2]+self.args.crop_size_d//2)]);
                    mask_c = torch.from_numpy(mask[:, int(center1[0]-self.args.crop_size_w//2):int(center1[0]+self.args.crop_size_w//2), int(center1[1]-self.args.crop_size_h//2):int(center1[1]+self.args.crop_size_h//2), int(center1[2]-self.args.crop_size_d//2):int(center1[2]+self.args.crop_size_d//2)]);

                # total_heatmap = torch.zeros_like(mri_c, dtype=torch.float64);

                # num_corrupted_patches = np.random.randint(1,5) if self.args.deterministic is False else 3;
                # for _ in range(0):
                #     mri_c, heatmap = add_synthetic_lesion_wm(mri_c, mask_c, self.args.deterministic)
                #     total_heatmap = torch.clamp(heatmap+total_heatmap, 0, 1);

                # total_heatmap_thresh = torch.where(total_heatmap > 0.5, 1.0, 0.0);
                # total_heatmap_thresh = torch.clamp(total_heatmap_thresh, 0, 1);

                
                total_heatmap_thresh = gt_c;

                pos_dt = distance_transform_edt(np.where(total_heatmap_thresh.squeeze().numpy()==1, 0, 1));
                pos_dt = pos_dt/(np.max(pos_dt)+1e-4);

                neg_dt = distance_transform_edt(total_heatmap_thresh.squeeze().numpy()==1);
                neg_dt = neg_dt/(np.max(neg_dt)+1e-4);

                dt = pos_dt - neg_dt ;
                dt = torch.from_numpy(np.expand_dims(dt, axis = 0));
                
                if self.args.debug_train_data:
                    pos_cords = np.where(total_heatmap_thresh >0.0);
                    if len(pos_cords[0]) != 0:
                        r = np.random.randint(0,len(pos_cords[0]));
                        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                    else:
                        center=[mri_c.shape[1]//2, mri_c.shape[2]//2, mri_c.shape[3]//2]
                    visualize_2d([mri_c, total_heatmap_thresh, mask_c + mri_c, gt_c], center);
                
                mri_c = self.transforms(mri_c);

                assert mri_c.shape[1] == 96 and mri_c.shape[2] == 96 and mri_c.shape[3] == 96

                if ret_mri is None:
                    ret_mri = mri_c.unsqueeze(dim=0);
                    ret_gt = total_heatmap_thresh.unsqueeze(dim=0);
                    ret_dt = dt.unsqueeze(dim=0);
                else:
                    ret_mri = torch.concat([ret_mri, mri_c.unsqueeze(dim=0)], dim=0);
                    ret_gt = torch.concat([ret_gt, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
                    ret_dt = torch.concat([ret_dt, dt.unsqueeze(dim=0)], dim=0);

        
            return ret_mri, ret_gt, ret_dt;
       
        else:
            mri, gt, loc, patient_id = self.data[index];

            mri = np.expand_dims(mri, axis=0);
            gt = np.expand_dims(gt, axis=0);

            mri = self.transforms(mri);


            if self.args.debug_train_data:
                pos_cords = np.where(gt == 1);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center=[mri.shape[1]//2, mri.shape[2]//2, mri.shape[3]//2]
                visualize_2d([mri, mask, gt], center);
            self.patient_id = patient_id;
            return mri, loc;
    def update_prediction(self, 
                          pred,  
                          loc):
        """saves the prediction into the predefined tensor.
            location is set through 'loc' parameter

        
        Parameters
        ----------
        pred : np.ndarray
            prediction from the model for a particular patch of MRI.

        patient_id : str
            indicate for which testing example this prediction is.

        loc : int
            location of this predicted patch in the list of all patches for one particular example.

        """
        self.pred_data[self.patient_id][(loc[0].item())*self.step_w:(loc[0].item())*self.step_w + self.args.crop_size_w, 
                                (loc[1].item())*self.step_h:((loc[1].item()))*self.step_h + self.args.crop_size_h, 
                                (loc[2].item())*self.step_d:((loc[2].item()))*self.step_d + self.args.crop_size_d] = np.array(pred.squeeze().detach().cpu().numpy()).astype("int32");

    def calculate_metrics(self, simple = True):
        """After finishing all the prediction, we calculate F1, HD and dice metrics

        Parameters
        ----------
        simple : bool
            if true, only dice score is computer, otherwise F1 and HD are also computed.
        """
        ret = [];
        for k in tqdm(self.pred_data.keys()):
            if simple is True:
                dice = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            else:
                dice,hd,f1 = calculate_metric_percase(self.pred_data[k].squeeze(), self.gt_data[k].squeeze(), simple=simple);
            if np.sum(self.gt_data[k].squeeze()) > 0:
                ret.append(dice if simple is True else [dice, hd, f1]);
        return np.mean(ret) if simple is True else np.mean(np.array(ret), axis =0);

def get_loader_pretrain_miccai(args):
    """prepare train and test loader for self-supervised pretraining model

        Parameters
        ----------
        args : dict
            arguments.
    """
    train_mri, test_mri = pickle.load(open(os.path.join('cache_miccai-2016', f'train_test_split.dmp'), 'rb'));

    mri_dataset_train = MICCAI_PRETRAIN_Dataset(args, train_mri[:1]);
    
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=args.num_workers, pin_memory=True);
    mri_dataset_test = MICCAI_PRETRAIN_Dataset(args, test_mri[:1], train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, test_loader; 

def get_loader_miccai16(args):
    """prepare train and test loader for new lesion segmentation model

        Parameters
        ----------
        args : dict
            arguments.
    """
    train_mri, test_mri = pickle.load(open(os.path.join('cache_miccai-2016', f'train_test_split.dmp'), 'rb'));


    mri_dataset_train = MICCAI_Dataset(args, train_mri if args.use_one_sample_only is False else train_mri[:1], train=True);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=args.num_workers, pin_memory=True);
    mri_dataset_test = MICCAI_Dataset(args, test_mri if args.use_one_sample_only is False else test_mri[:1], train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, test_loader, mri_dataset_test; 

def visualize_2d(images, slice,):
    """display one slice of an MRI scan, for debugging purposes

        Parameters
        ----------
        args : dict
            arguments.
    """
    fig, ax = plt.subplots(len(images),3);
    for i,img in enumerate(images):
        img = img.squeeze();
        ax[i][0].imshow(img[slice[0], :,:], cmap='gray');
        ax[i][1].imshow(img[:,slice[1],:], cmap='gray');
        ax[i][2].imshow(img[:,:,slice[2]], cmap='gray');
    plt.show()

def add_synthetic_lesion_wm(img, 
                            mask_g, 
                            deterministic):
    """adds synthetic lesions to the MRI scan

        Parameters
        ----------
        img : np.ndarray
            MRI scan patch

        mask_g : np.ndarray
            mask to take inpainting centers from

        deterministic : bool
            if true, every call to this function yields the same results.
    """
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(10,20) if deterministic is False else 3;
    size_y = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    size_z = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    mask_cpy[:,:,:,d-size_z:] = 0;
    mask_cpy[:,:,:,:size_z+1] = 0;
    mask_cpy[:,:,w-size_y:,:] = 0;
    mask_cpy[:,:,:size_y+1,:] = 0;
    mask_cpy[:,h-size_x:,:,:] = 0;
    mask_cpy[:,:size_x+1,:,:] = 0;
    pos_cords = np.where(mask_cpy==1);

    if deterministic is False:
        if len(pos_cords[0]) != 0:
            r = np.random.randint(0,len(pos_cords[0]));
            center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    else:
        if len(pos_cords[0]) != 0:
            center = [pos_cords[1][int(len(pos_cords[0])//2)], pos_cords[2][int(len(pos_cords[0])//2)],pos_cords[3][int(len(pos_cords[0])//2)]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    
 
    #shape
    cube = torch.zeros((1,h,w,d), dtype=torch.uint8);
    cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), max(center[1]-size_y,0):min(center[1]+size_y,w), max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;
    cube = cube * mask_g;

    cube = GaussianSmooth(1.2, approx='erf')(cube);
    cube = cube / (torch.max(cube) + 1e-4);
    #================

    noise = (torch.ones((1,h,w,d), dtype=torch.uint8));
    final = (cube)*(noise);
    mri_after = (1-cube)*mri + final;
    
    return mri_after, cube;