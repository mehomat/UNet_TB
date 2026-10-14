import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io, transform, filters
import cv2

class custom_loader_training(Dataset):
 
    """
    Dataset loader for training images from Elf lab pipeline.
    args: 
        phase_ims = list of phase contrast directories as full path: path/to/image/im.tiff
        mask_ims = list of masks to train segmentation as full path: path/to/mask/mask.tiff
        important! the list must be in the same order.
        """
    
    def __init__(self, phase_ims, mask_ims, scales=None, transform=None):
        self.phase_ims = phase_ims
        self.mask_ims = mask_ims
        self.scales = scales
        self.transform = transform
            
    def __len__(self):
        return len(self.phase_ims)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #send to float32 type, PyTorch supposedly likes it
        phase_im = io.imread(self.phase_ims[idx],as_gray=True)#.astype('float32')
        mask_im = io.imread(self.mask_ims[idx],as_gray=True).astype('float32') #albumentation needs 32bit
        
        # sample = {'im' : phase_im, 'mask' : mask_im}
        #print(f'Initial image min and max: {phase_im.min(), phase_im.max()}')
        if self.scales is not None and self.scales[idx] != 1.0:
            s = self.scales[idx]
            h, w = phase_im.shape[:2]
            size = (round(w * s), round(h * s))
            phase_im = cv2.resize(phase_im, size, interpolation=cv2.INTER_AREA)
            mask_im = cv2.resize(mask_im, size, interpolation=cv2.INTER_NEAREST)
        
        if self.transform is not None:
            transformed = self.transform(image = phase_im, mask = mask_im)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = phase_im
            mask = mask_im
        #print(f'Transfored image min and max: {image.min(), image.max()}')
        return image, mask
        
class custom_loader_segmentation(Dataset):
    
    """Dataset loader for segmentation images from Elf lab pipeline.
    args: 
        phase_ims = list of phase contrast directories as full path: path/to/image/im.tiff
    returns normalised image and a path to save image to"""
    
    def __init__(self, phase_ims, output_folder, transform=None):

        # assert 'PreprocessedPhase' in phase_ims[0] #check if the phase images have been preprocessed before analysis

        self.phase_ims = phase_ims
        self.transform = transform
        self.output_folder = output_folder
            
    def __len__(self):
        return len(self.phase_ims)
    
    def __getitem__(self, idx):
        
        #i dont know what it does but it was there in tutorial
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #send to float32 type, PyTorch supposedly likes it
        # phase_im = io.imread(self.phase_ims[idx], as_gray=True).astype('uint16')
        phase_im = io.imread(self.phase_ims[idx], as_gray=True).astype('float32')
        
        path_to_save = self.phase_ims[idx].replace('PreprocessedPhase', self.output_folder)
        path_to_save = path_to_save.replace('.tiff', '_KubNet.tiff')
   
        if self.transform:
            phase_im = self.transform(phase_im)
            
        return {'im' : phase_im, 'path_to_save' : path_to_save}