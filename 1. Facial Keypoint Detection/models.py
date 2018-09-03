## DONE: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## DONE: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) 
        # to avoid overfitting
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4)       
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2) 
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2) 
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32) 
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.5)
            
        self.fc1 = nn.Linear(64*11*11, 512)      
        self.fc2 = nn.Linear(512, 136)        
        
    def forward(self, x):
        ## DONE: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.bn1(F.leaky_relu(self.conv1(x), 0.1))     
        x = self.bn2(F.leaky_relu(self.conv2(x), 0.1))            
        x = self.bn3(F.leaky_relu(self.conv3(x), 0.1)) 
        x = self.drop1(x)
        
        x = x.view(x.size(0), -1)
           
        x = self.bn4(self.fc1(F.elu(x)))
        x = self.drop2(x)        
        x = self.fc2(F.elu(x))        

        return x