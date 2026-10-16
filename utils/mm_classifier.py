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
    
    def __init__(self, net, optimizer, scheduler, num_epochs, save_dir, net_name, val_every=10):
        """
        The classifier for mm experiment
            net (nn.Module): The neural net module containing the definition of your model
            num_epochs (int): The maximum number of epochs on which the model will train
            val_every (int): validate every val_every epochs and after the last epoch
        """
        self.net = net
        self.epoch_counter = 0
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.train_losses = []
        self.val_every = val_every
        self.val_epochs = []
        self.val_losses = []
        self.val_dices = []
        self.best_val_loss = np.inf
        self.save_name = save_dir + "/models/" + net_name
        os.makedirs(save_dir + "/models", exist_ok=True)

    def loss_function(self, pred, target):
        #define loss as BCE + DC
        pred = torch.sigmoid(pred)

        num = target.size(0)
        m1 = pred.view(num,-1)
        m2 = target.view(num,-1)
        # DEBUG — remove after fix is confirmed
        #print(f"m1 min: {m1.min():.4f}, max: {m1.max():.4f}")#, unique vals: {torch.unique(m2[:10])}")

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
        
        epoch_loss = float(np.mean(running_losses))
        print(f"loss: {epoch_loss}")
        return epoch_loss

    def validate_epoch(self, validation_loader):
        #run network on all validation images, return mean loss and mean Dice (threshold 0.5)
        val_losses, dices = [], []
        with torch.no_grad():
            for i_batch, (im, target) in enumerate(validation_loader):
                
                im = im.cuda()
                target = target.cuda()
                res = self.net(im)
                val_losses.append(self.loss_function(res, target).item())

                pred = torch.sigmoid(res) > 0.5
                gt = target > 0.5
                denom = (pred.sum() + gt.sum()).item()
                dices.append(1.0 if denom == 0 else 2. * (pred & gt).sum().item() / denom)

                #save the prediction for the first validation image to follow training visually
                if i_batch == 0:
                    os.makedirs(self.save_dir + '/Validation', exist_ok=True)
                    file_name_seg = 'epoch_' + str(self.epoch_counter+1) + '_seg.tiff'
                    io.imsave(self.save_dir + '/Validation/' + file_name_seg,
                              torch.sigmoid(res).cpu().numpy().squeeze(), check_contrast=False)

        val_loss, val_dice = float(np.mean(val_losses)), float(np.mean(dices))
        print(f"validation loss: {val_loss}, validation Dice: {val_dice:.4f}")
        return val_loss, val_dice
        
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
        train_loss = self.train_epoch(train_loader)

        self.train_losses.append(train_loss)

        epoch = self.epoch_counter + 1
        if len(validation_loader) == 0 or (epoch % self.val_every != 0 and epoch != self.num_epochs):
            return

        self.net.eval()
        val_loss, val_dice = self.validate_epoch(validation_loader)
        # self.validate_custom()

        self.val_epochs.append(epoch)
        self.val_losses.append(val_loss)
        self.val_dices.append(val_dice)

        #keep the checkpoint with the lowest validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'model_state_dict': self.net.state_dict()},
                       self.save_name.replace('.pth', '_best.pth'))
        
    def train(self, train_loader: DataLoader, validation_loader: DataLoader):
        
        #call training of the network
        if len(validation_loader) == 0:
            print("no validation images, skipping validation; use the final model")
        for epoch in range(self.num_epochs):
            self.run_epoch(train_loader, validation_loader)
            
            if self.scheduler:
                self.scheduler.step()

            self.epoch_counter = self.epoch_counter + 1
        
            # if (epoch+1)%10 == 0:
            #     this_save_name = f"{self.save_name[:-4]}_{epoch+1}.pth"
            #     torch.save({
            #             'epoch': self.epoch_counter,
            #             'model_state_dict' : self.net.state_dict(),
            #             'optimizer_state_dict' : self.optimizer.state_dict()
            #             },  this_save_name)
            
            # torch.save({
            #         'epoch': self.epoch_counter,
            #         'model_state_dict' : self.net.state_dict(),
            #         'optimizer_state_dict' : self.optimizer.state_dict()
            #         },  self.save_name)
        
        plt.figure()
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        plt.plot(self.val_epochs, self.val_losses, 'o-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        #plt.show()

        #self.save_name
        plt.savefig(self.save_name.replace('.pth', '_loss_curve.png'))
        #save the final model and the loss history
        torch.save({'model_state_dict' : self.net.state_dict(),
                    'train_losses': self.train_losses,
                    'val_epochs': self.val_epochs,
                    'val_losses': self.val_losses,
                    'val_dices': self.val_dices},  self.save_name)
             