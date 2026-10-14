import os
import re
import warnings

def getFileList(path_to_file, chan_name = ''):
    """
    supporting function to walk in the expriment directory and return a 
    list of the all files existing in the master directory. 
    For PraNet segmentation of RiteAcquire experiments. 
    
    arg: chan = 'cbright', to extract images from only phase contrast.
    
    20200410 Kuba
    """   
    file_list = []
    for (root,_,files) in os.walk(path_to_file):
        for name in files:
            if re.search(chan_name,root) is not None:
                if name.endswith(('.tiff','.tif')):
                    file_list.append(os.path.join(root,name))
    if len(file_list) == 0:
        print(f"No images found in {path_to_file}")
    
    return sorted(file_list)

def makeDirs(exp_dir, save_dir):
    """"supporting function, making folders to put the segmentation images into
    args:
    destingation_dir = direction to the experiment directory
    save_dir = SegmentedPhase, or SegmentedChans

    20200424 Kuba
    """
    for (root,_,_) in os.walk(exp_dir):
        if re.search('PreprocessedPhase',root) is not None:
            seg_dir = root.replace('PreprocessedPhase',save_dir)
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)

def getTrainingPairs(training_dirs):
    """
    Pair phase images with their masks by file name, not by sorted order
    (sorting breaks with mixed-width numbers, e.g. _10.tiff, _100.tiff, _10_masks.tiff).
    Supported names: X_img.tif / X_mask.tif and X.tiff / X_masks.tiff.
    Raises if an image has no mask or a mask has no image.
    """
    mask_suffix = re.compile(r'_masks?$')
    phase, mask = [], []
    for dr, nm in training_dirs:
        images, masks = {}, {}
        for f in getFileList(dr, nm):
            stem = os.path.splitext(f)[0]
            if mask_suffix.search(stem):
                masks[mask_suffix.sub('', stem)] = f
            else:
                images[re.sub(r'_img$', '', stem)] = f
        unpaired = sorted(set(images) ^ set(masks))
        if unpaired:
            raise ValueError(f'{dr}: {len(unpaired)} files without a partner, e.g. {unpaired[:3]}')
        for key in sorted(images):
            phase.append(images[key])
            mask.append(masks[key])
    return phase, mask

def main():
    pass

if __name__ == '__main__':
    main()