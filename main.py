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

from utils.cell_aware_crop import CellAwareCrop

def train_net(train_dirs, save_dir, net_size=256, training_transform=None, normalize=custom_normalize99, num_epochs=120, batch_size=4, scheduler=None, optimizer=None):

    #load the data
    phase, mask, scales = [], [], []
    for d, nm, s in train_dirs:
        p, m = getTrainingPairs([(d, nm)])
        phase += p
        mask += m
        scales += [s] * len(p)

    print(f'Number of phase images: {len(phase)}')
    print(f'Number of binary masks : {len(mask)}')

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

    validate_dataset = custom_loader_training(phase_ims = phase,
                                              mask_ims = mask,
                                              scales  = scales,
                                              transform = A.Compose([A.PadIfNeeded(min_height=None, min_width=None,
                                                                                   pad_height_divisor=16, pad_width_divisor=16,
                                                                                   border_mode=0),   # zero-pad to next multiple of 16
                                                                    A.Lambda(name='normalize', image=normalize, p=1.0),
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

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d_%H%M%S')
    net_name = f'unetTB_{net_size}_'+ts+'.pth'
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