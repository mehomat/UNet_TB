import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tqdm import tqdm

from skimage.measure import label
from skimage import io

import matplotlib.pyplot as plt

import os
import numpy as np

class mm_classifier:
    
    """ trainer for microscopy experiment UNet. 
    The classifier will run 1 epoch, then test the accuracy on test data.
    
    this class is designed to work with albumentations library
    """
    
    def __init__(self, net, optimizer, scheduler, num_epochs, save_name):
        """
        The classifier for mm experiment
            net (nn.Module): The neural net module containing the definition of your model
            num_epochs (int): The maximum number of epochs on which the model will train
        """
        self.net = net
        self.epoch_counter = 0
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_name = save_name

    def loss_function(self, pred, target):
        #define loss as BCE + DC
        #print(f'pred max value:{torch.max(pred)}, min value: {torch.min(pred)}')
        #release from logits
        pred = torch.sigmoid(pred)

        num = target.size(0)
        m1 = pred.view(num,-1)
        m2 = target.view(num,-1)
        #print(f'm1 max value:{torch.max(m1)}, min value: {torch.min(m1)}')
        #print(f'm2 max value:{m2.max()}, min value: {m2.min()}')
        
        # binary cross entropy 
        BCE = F.binary_cross_entropy(m1,m2)
        
        #try only binary cross entropy
        # return BCE

        #dice loss
        intersection = (m1 * m2)
        DC = 2. * (intersection.sum(1)) / (m1.sum(1) + m2.sum(1))
        DC = 1 - DC.sum() / num

        return DC + 0.5*BCE
            
    def train_epoch(self, train_loader):
        
        print(f"training epoch {self.epoch_counter+1}")
        running_losses = []

        for i_batch, (im, target) in enumerate(train_loader):
            #designed to work with albumentations library
            
            im = im.cuda() 

            self.optimizer.zero_grad()
            loss=0

            output = self.net(im)

            loss = self.loss_function(output, target.cuda())
            
            running_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        
        print(f"loss: {np.mean(running_losses)}")

    def validate_epoch(self, validation_loader):
        
        #run network on validation data after each epoch.
        for i_batch, (im, target) in enumerate(validation_loader):
            
            im = im.cuda() 
            res = self.net(im)

            res = torch.sigmoid(res)
            res = res.to("cpu").detach().numpy().squeeze(0).squeeze(0)
            
            file_name_seg = 'epoch_' + str(self.epoch_counter+1) + '_' + str(i_batch) + '_seg.tiff'
            io.imsave('/home/spartak/elflab/BSL3/analysis/EXP-26-CB9767/Validation/' + file_name_seg, res)
            
            if i_batch > 0:
                torch.cuda.empty_cache()
                break        
        
    def validate_custom(self):
        #small validation on the fixed set of real experimental data, 2 for mother machine, 2 for giant chip, no agarose pad yet.
        #some hardcoded values here - move them outside if necessary
        dir_name = '/home/skynet/code/UNet_testCases/'
        test_cases = os.listdir(dir_name)
        namespace = 'ABCDEFGHIJKLMNOP'
        #load, normalize and predict for each image in the test directory
        for fi, f in enumerate(test_cases):
            
            im = io.imread(os.path.join(dir_name, f))
            im = im.astype('float32')
            im = torch.from_numpy(im)
            im = im.unsqueeze(0).unsqueeze(0)
            im = im.cuda()
            im = (im - torch.mean(im)) / torch.std(im)
            res = self.net(im)
            res = torch.sigmoid(res)
            res = res.to("cpu").detach().numpy().squeeze(0).squeeze(0)
            f_out = f"{namespace[fi]}_{str(self.epoch_counter)}.tiff"
            io.imsave('/hdd/RecPAIR/PraNetTraining/' + f_out, res)

    def run_epoch(self, train_loader: DataLoader, validation_loader: DataLoader, callbacks=None):
        # run a single epoch and validate the output.
        
        self.net.train()
        self.train_epoch(train_loader)

        self.net.eval()
        self.validate_epoch(validation_loader)
        # self.validate_custom()
        
    def train(self, train_loader: DataLoader, validation_loader: DataLoader):
        
        #call training of the network
        for epoch in range(self.num_epochs):
            self.run_epoch(train_loader, validation_loader)
            
            if self.scheduler:
                self.scheduler.step()

            self.epoch_counter = self.epoch_counter + 1
        
            if (epoch+1)%10 == 0:
                this_save_name = f"{self.save_name[:-4]}_{epoch+1}.pth"
                torch.save({
                        'epoch': self.epoch_counter,
                        'model_state_dict' : self.net.state_dict(),
                        'optimizer_state_dict' : self.optimizer.state_dict()
                        },  this_save_name)
            
            torch.save({
                    'epoch': self.epoch_counter,
                    'model_state_dict' : self.net.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict()
                    },  self.save_name)
        
        
        #save only the model
        # torch.save({'model_state_dict' : self.net.state_dict()},  self.save_name)
             