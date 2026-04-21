from __future__ import absolute_import, division
import numpy as np
import torch

#to be used with albumentations lambda transformation
def custom_normalize(image, **kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    mean = np.mean(image)
    std = np.std(image)
    if std < 1e-6:                          # guard against flat/empty patches
        return np.zeros_like(image, dtype=np.float32)
    image = ((image - mean) / std).astype(np.float32)
    return image

def custom_to_tensor(image, **kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    this_type = torch.FloatTensor
    # this_type = torch.HalfTensor #16-bit floating point
    image = torch.from_numpy(image).type(this_type)
    image = image.unsqueeze(0)
    return image

def custom_gauss_noise(image,**kwargs):
    #HANDLE TO A CUSTOM NORMALIZATION FUNCTION TO BE USED WITH ALBUMENTATION LIBRARY LABMDA TRANSFORMATION FUNCTION
    # For 8-bit images (0-255): sigma 2-20 is realistic microscopy noise
    sigma = np.random.uniform(2, 20)
    gauss = np.random.normal(0, sigma, image.shape)
    #print(f"image range: [{image.min()}, {image.max()}], noise range: [{gauss.min()}, {gauss.max()}]")
    return image + gauss

    return image + gauss

def main():
    pass

if __name__ == '__main__':
    main()