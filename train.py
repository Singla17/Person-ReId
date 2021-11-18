# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:01:06 2021

@author: Somanshu
"""
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

import argparse
import os
import time
import random

from model import LATransformer
import timm
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def parse_args():
    """
    Returns the arguments parsed from command line
    """
    parser = argparse.ArgumentParser(description='Getting various paths')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the model weights")
    parser.add_argument('-c', '--num_classes', type=int, default=62, required=True, \
                                                        help="Number of classes in trainset")
    
        
    args = parser.parse_args()
    return args


def set_seed(seed):
    """
    This method helps to seed the libraries, it is important to get reproducible results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def train_one_epoch(epoch, model, loader, optimizer, loss_fn,verbose=False):
    """
    This method implements training of model for one epoch
    """
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    end = time.time()

    for index,(data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()
        output = model(data)
        score = 0.0
        sm = nn.Softmax(dim=1)
        for k, v in output.items():
            score += sm(output[k])
        _, preds = torch.max(score.data, 1)
        
        loss = 0.0
        for k,v in output.items():
            loss += loss_fn(output[k], target)
        loss.backward()

        optimizer.step()

        batch_time_m.update(time.time() - end)
        acc = (preds == target.data).float().mean()

        epoch_loss += loss/len(loader)
        epoch_accuracy += acc / len(loader)

    if verbose:
        print("The loss at epoch "+str(epoch)+ " was "+str(epoch_loss.data.item())+ " and the training accuracy is "+str(epoch_accuracy.data.item()))

    return OrderedDict([('train_loss', epoch_loss.data.item()), ("train_accuracy", epoch_accuracy.data.item())])

def visualization(loss_arr,epo,loss):
  """
  This is to visualize the training curves
  """
  x = np.linspace(1,epo,epo)
  if loss:
      plt.plot(x,loss_arr, label='Training Loss')
      plt.ylabel('Loss')
      plt.xlabel('Epochs')
      plt.title('Training Curve')
  else:
      plt.plot(x,loss_arr, label='Training Accuracy')
      plt.ylabel('Accuracy')
      plt.xlabel('Epochs')
      plt.title('Training Curve')
  
  plt.legend()
  plt.show()


def training(model,optimizer,criterion,scheduler,num_epochs,verbose,blocks,unfreeze_after,train_loader):
    """
    Simulates the training of model
    """
    unfrozen_blocks = 0
    train_loss=[]
    train_accuracy=[]
    
    for epoch in range(num_epochs):
        if epoch%unfreeze_after==0:
            unfrozen_blocks += 1
            model = unfreeze_blocks(model,blocks, unfrozen_blocks)
            optimizer.param_groups[0]['lr'] *= lr_decay 
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Unfrozen Blocks: {}, Current lr: {}, Trainable Params: {}".format(unfrozen_blocks, 
                                                                                 optimizer.param_groups[0]['lr'], 
                                                                                 trainable_params))
    
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, criterion,verbose=verbose)
        train_loss.append(train_metrics["train_loss"])
        train_accuracy.append(train_metrics["train_accuracy"])
    
    visualization(train_loss, num_epochs, True)
    visualization(train_accuracy, num_epochs, False)
    
    return model

def freeze_all_blocks(model,blocks):
    """
    This method is used to freeze all 12[as per the original publication] blocks of a ViT
    """
    frozen_blocks = blocks
    for block in model.model.blocks[:frozen_blocks]:
        for param in block.parameters():
            param.requires_grad=False
            
            
def unfreeze_blocks(model, blocks, amount= 1):
    """
    This method is used to unfreeze some blocks of the pretrained-ViT
    """
    for block in model.model.blocks[(blocks-1)-amount:]:
        for param in block.parameters():
            param.requires_grad=True
    return model


if __name__ == "__main__":
    
    """
    Setting up input and output paths
    """
    args = parse_args()
    inp_path = args.inp_path
    out_path = args.out_path
    num_classes = args.num_classes
    
    """
    Checks the availibility of a GPU
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using {}".format(device))
    set_seed(0)
    
    """
    Hyper-parameters
    """
    batch_size = 32
    num_epochs = 30
    lr = 3e-4
    gamma = 0.7
    unfreeze_after=2
    lr_decay=.8
    lmbd = 8
    
    """
    Input Images can be of various sizes so we define some resizing and normalization
    parameters 
    """
    transform_train_list = [
    transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    
    data_transforms = {
    'train': transforms.Compose( transform_train_list )
    }
    
    
    """
    Loading the base ViT model.
    """
    vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    vit_base= vit_base.to(device)
    vit_base.eval()
    
    """
    Loading the data
    """
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(inp_path, 'train'),
                                          data_transforms['train'])
    
    train_loader = DataLoader(dataset = image_datasets['train'], batch_size=batch_size, shuffle=True )


    """
    Setting up model and training
    """
    # Create LA Transformer
    num_la_blocks = 14  ## Number of locally aware classifiers in the training model.
    BLOCKS = 12  ## Not to be touched
    INT_DIM = 768   ## Latent Vector Size [internal to ViT]  ## Not to be touched
    model = LATransformer(vit_base, lmbd,num_classes,num_la_blocks,BLOCKS,INT_DIM).to(device)
    print(model.eval())
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(),weight_decay=5e-4, lr=lr)
    
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    freeze_all_blocks(model,BLOCKS)
    
    """
    Training Begins
    """
    
    print("Training Begins...")
    model = training(model,optimizer,criterion,scheduler,num_epochs,True,BLOCKS,unfreeze_after,train_loader)
    print("Training Completed")
    torch.save(model.cpu().state_dict(), out_path)
        
    