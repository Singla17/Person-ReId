import torch
import torch.nn as nn

## Acknowledgement: this code is adapted from https://github.com/ppriyank/Pytorch-Additive_Margin_Softmax_for_Face_Verification

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size,)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


        

class AM_Softmax(nn.Module): #requires classification layer for normalization 
    """
    Implements the AM softmax loss function 
    """
    def __init__(self, m=0.35, s=30, d=2048, num_classes=625, use_gpu=True , epsilon=0.1,smoothing=True):
        super(AM_Softmax, self).__init__()
        self.m = m
        self.s = s 
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes , use_gpu=use_gpu)
        self.smoothing = smoothing

    def forward(self, features, labels , classifier  ):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension 
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class, assumed to be an object of nn.sequential
        '''
        features = nn.functional.normalize(features, p=2, dim=1) # normalize the features
        with torch.no_grad():
            classifier[0].weight=nn.Parameter(classifier[0].weight.div(torch.norm(classifier[0].weight, dim=1, keepdim=True)))  ## [0] bracing as assumed to be from nn.sequential

        cos_angle = classifier(features)
        cos_angle = torch.clamp( cos_angle , min = -1 , max = 1 ) 
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]]  - self.m 
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle , labels, use_label_smoothing=self.smoothing)
        return log_probs
"""
loss_f = CrossEntropyLabelSmooth(5,epsilon=0.2,use_gpu=False)
import torch 
import numpy as np
labels = torch.from_numpy(np.array([0,2,4])).long()
op= torch.randn( 3,5)
loss = loss_f.forward(op,labels,use_label_smoothing=False)
print(loss)
"""