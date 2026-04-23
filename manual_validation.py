from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import prune
from torchvision import transforms
import torch

import os

from utils.unet import UNet, UNet_deep
from utils.watershed import watershed

from skimage import io, measure, morphology, feature, color, transform, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

from os import walk

def segment():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = UNet_shallow(max_filters = 512)

    # net = UNet_deep()
    # NET_PATH = '/hdd/RecPAIR/UNet_deep_universal_2021_11_19.pth'

    # net = UNet(max_filters = 512)
    # NET_PATH = '/hdd/RecPAIR/UNet_normal_universal_2021_12_06.pth'

    net = UNet(max_filters = 512)
    NET_PATH = '/hdd/RecPAIR/UNet_normal_growthChannels_2021_12_07.pth'


    saved_net = torch.load(NET_PATH)
    net.load_state_dict(saved_net['model_state_dict'])
    net.eval()
    net.cuda()

    dir_name = '/home/skynet/code/UNet_testCases/'
    test_cases = os.listdir(dir_name)
    # dir_name = '/hdd/05 El330/O2/Pos3/phase'

    # test_cases = ['img_000000000.tiff']

    #load, normalize and predict for each image in the test directory
    for fi, f in enumerate(test_cases):
        print(os.path.join(dir_name, f))
        
        im = io.imread(os.path.join(dir_name, f))
        im_org = io.imread(os.path.join(dir_name, f))
        
        sz = im.shape
        pad_with = np.ceil(np.array(sz)/16)*16 - sz
        pad_with = pad_with.astype('int')
        im = np.pad(im, pad_width=((0,pad_with[0]),(0,pad_with[1])),mode='constant')

        im = im.astype('float32')
        im = (im - np.mean(im)) / np.std(im)
        im = torch.from_numpy(im)
        im = im.unsqueeze(0).unsqueeze(0)
        im = im.cuda()
        res = net(im)
        res = torch.sigmoid(res)
        res = res.to("cpu").detach().numpy().squeeze(0).squeeze(0)
        
        res = res[0:sz[0],0:sz[1]]

        res = res > 0.5
        
        res = morphology.remove_small_objects(res,50)
        # res = watershed(res)
        res = res > 0
        outlines = feature.canny(res)

        
        plt.figure()

        vmin = np.percentile(im_org,5)
        vmax = np.percentile(im_org,95)
        
        im_org = color.gray2rgb(im_org)
        im_org = im_org - np.min(im_org)
        im_org = im_org / np.max(im_org)

        for yi,y in enumerate(outlines):
            for xi,x in enumerate(y):
                if x:
                    im_org[yi][xi] = (0,1,0)

        plt.imshow(im_org,cmap=plt.cm.gray)
        plt.show()
        # res = img_as_ubyte(res)
        # io.imsave(os.path.join(dir_name, f).replace('.','_segmented.'),res,compress=6)

def saveBack():
    
    net = UNet_deep()
    NET_PATH = '/hdd/RecPAIR/UNet_deep_universal_2021_11_19.pth'
    n = torch.load(NET_PATH)
    model=dict()
    model['model_type'] = 'UNet_deep'
    model['model_state_dict'] = n
    model['arhc'] = net
    # torch.save(model,'/hdd/RecPAIR/UNet_uni_pipeline_backwardsCompatibility.pth', _use_new_zipfile_serialization=False)
    torch.save(model,'/hdd/RecPAIR/UNet_deep_mdma.pth', _use_new_zipfile_serialization=True)

def segmentTestExp():

    #segment a whole bunch of images from a single experiment

    dir = '/home/spartak/elflab/BSL3/analysis/EXP-26-CB9767/KI_data'

    res_dir = '/home/spartak/elflab/BSL3/analysis/EXP-26-CB9767/KI_data_seg'

    images = [f"{dir}/{x}" for x in os.listdir(dir)]
    images.sort()
    images = images[:-1]
    images = images[::2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet(max_filters = 512)

    NET_PATH = '/home/spartak/elflab/BSL3/analysis/EXP-26-CB9767/models/UNet_TB_512_Adam_252026-04-22_12:15:01_120.pth'

    saved_net = torch.load(NET_PATH)
    net.load_state_dict(saved_net['model_state_dict'])
    net.eval()
    net.cuda()

    # #load, normalize and predict for each image in the test directory
    with torch.no_grad():
        for fi, f in enumerate(images):
            
            im_org = io.imread(f)
            
            xorg,yorg = np.shape(im_org)
            x,y = np.shape(im_org)
            xpad, ypad = 0, 0
            if x%16 != 0:
                xpad = 16-(x%16)
            if y%16 != 0:
                ypad = 16-(y%16)
            
            im_org = np.pad(im_org, ((0,xpad),(0,ypad)),'constant',constant_values=(0,0))
            
            im = im_org.astype('float32')
            im = (im - np.mean(im)) / np.std(im)
            im = torch.from_numpy(im)
            im = im.unsqueeze(0).unsqueeze(0)
            im = im.cuda()
            res = net(im)
            res = torch.sigmoid(res)
            res = res.to("cpu").detach().numpy().squeeze(0).squeeze(0)
            
            res = res[0:xorg,0:yorg]
            res = res > 0.5
            #res = morphology.remove_small_objects(res,50)

            #res = measure.label(res)

            res = res.astype('uint8')
            res[res==1] = 255
            #props = measure.regionprops(res)
            
            #mat = np.zeros(np.shape(res))

            #reconstruct the image only using centroids
            #for identity, p in enumerate(props):
            #    x,y = np.round(p.centroid)
            #    mat[int(x),int(y)] = identity

            #res = morphology.dilation(mat, morphology.disk(3,dtype='uint16'))
            #res = res.astype('uint16')
            filename = f.split('/')[-1]
            io.imsave(os.path.join(res_dir,filename),res,check_contrast=False)

def main():
    #segment()
    # saveBack()
    segmentTestExp()
    
if __name__ == '__main__':
    main()
    

