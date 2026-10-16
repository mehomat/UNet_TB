from torch.utils.data import DataLoader
#from torchvision import transforms
import torch

from utils.custom_loader import custom_loader_training
from utils.mm_classifier import mm_classifier
from utils.custom_a_handles import custom_normalize, custom_to_tensor, custom_gauss_noise, custom_normalize99

#using albumentations library
import albumentations as A
#import albumentations.augmentations.functional as F

# from utils.unet import UNet
from utils.unet import UNet

#from skimage import io
#import cv2
#import matplotlib.pyplot as plt
import numpy as np

from utils.load_files import getTrainingPairs

from sklearn.model_selection import train_test_split
import time, datetime
import os, json

from utils.cell_aware_crop import CellAwareCrop

def train_net(train_dirs, save_dir, net_size=256, training_transform=None, normalize=custom_normalize99, num_epochs=120, batch_size=4, scheduler=None, optimizer=None, val_fraction=0.2, seed=0):
    """
    train_dirs entries: (dir, chan_name, scale) or (dir, chan_name, scale, val_fraction).
    Each dataset is split separately into train/validation; val_fraction=0 puts all of it into training.
    """
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d_%H%M%S')
    net_name = f'unetTB_{net_size}_'+ts+'.pth'

    #load the data and split each dataset separately
    phase, mask, scales = [], [], []
    val_phase, val_mask, val_scales = [], [], []
    for entry in train_dirs:
        d, nm, s = entry[:3]
        frac = entry[3] if len(entry) > 10 else val_fraction
        p, m = getTrainingPairs([(d, nm)])
        if frac > 0:
            p, vp, m, vm = train_test_split(p, m, test_size=frac, random_state=seed)
        else:
            vp, vm = [], []
        phase += p
        mask += m
        scales += [s] * len(p)
        val_phase += vp
        val_mask += vm
        val_scales += [s] * len(vp)
        print(f'{d}: {len(p)} train, {len(vp)} validation')

    print(f'Number of training images  : {len(phase)}')
    print(f'Number of validation images: {len(val_phase)}')

    #save the split next to the model so it can be reused for model comparisons
    os.makedirs(save_dir + '/models', exist_ok=True)
    with open(save_dir + '/models/' + net_name.replace('.pth', '_split.json'), 'w') as f:
        json.dump({'train': phase, 'validation': val_phase, 'seed': seed}, f, indent=1)

    #specify transformations and unets
    if training_transform is None:
        crop_window = min(net_size,400)
        training_transform = A.Compose(
            [
                #A.RandomRotate90(p=1),
                A.RandomCrop(crop_window, crop_window, p=1),
                #CellAwareCrop(p=1,crop_size=crop_window, min_cell_fraction=0.05, max_attempts=20),
                A.GridDistortion(p=0.4),
                #A.ElasticTransform(p=0.4, alpha=50, sigma=20),
                A.Affine(p=0.8, translate_percent=(-0.2, 0.2), scale=(0.75, 1.25), rotate=(-5, 5)),
                A.GaussianBlur(p=0.6, blur_limit=0, sigma_limit=(0.1, 2.0)),
                A.Lambda(name='gauss-noise', image=custom_gauss_noise, p=0.5),
                A.GridDropout(p=0.3, ratio=0.3, random_offset=True),
                #A.Lambda(name='normalize', image=custom_normalize, p=1.0),
                A.Lambda(name='normalize', image=normalize, p=1.0),
                A.Lambda(name='to_tensor', image=custom_to_tensor, mask=custom_to_tensor, p=1.0)
            ]
            )

    training_dataset = custom_loader_training( phase_ims = phase,
                                               mask_ims  = mask,
                                               scales  = scales,
                                               transform = training_transform)

    validate_dataset = custom_loader_training(phase_ims = val_phase,
                                              mask_ims = val_mask,
                                              scales  = val_scales,
                                              transform = A.Compose([
                                                                    A.Lambda(name='normalize', image=normalize, p=1.0),
                                                                    A.PadIfNeeded(min_height=None, min_width=None,
                                                                                   pad_height_divisor=16, pad_width_divisor=16,
                                                                                   border_mode=0),   # zero-pad to next multiple of 16
                                                                    A.Lambda(name='to_tensor', image=custom_to_tensor, mask=custom_to_tensor, p=1.0),
                                                                    ]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet(max_filters = net_size)
    net.to(device)
    
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers = 10)
    validation_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False)

    #optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
    if optimizer is None:
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.3,patience=5,verbose=True)

    classifier = mm_classifier(net=net, 
                               optimizer = optimizer, 
                               scheduler = scheduler, 
                               num_epochs = num_epochs, 
                               save_dir = save_dir,
                               net_name = net_name
                               )

    classifier.train(training_loader, validation_loader)
 
def main():
    #fill in the name
    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    train = 'traps'
    net_size = 400#256
    if train == 'cells':
        NET_NAME = f'unetTB{net_size}_'+ts
    else:
        NET_NAME = f'unetTB{net_size}_traps_'+ts
    save_name = f"/home/spartak/elflab/BSL3/analysis/EXP-26-CB9769/models/{NET_NAME}.pth"
    train_net(save_name,train=train, net_size=net_size)

if __name__ == '__main__':
    main()