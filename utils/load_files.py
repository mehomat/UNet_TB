import os
import re

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
    images = [getFileList(dr,nm) for dr, nm in training_dirs]
    images = [this_phase for each_set in images for this_phase in each_set]
    phase = images[::2]
    mask = images[1::2]
    return phase, mask

def main():
    pass

if __name__ == '__main__':
    main()