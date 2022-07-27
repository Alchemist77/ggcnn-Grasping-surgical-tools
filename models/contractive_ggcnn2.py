import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt
from torch.autograd import Variable

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]
fig = plt.figure()
rows = 2
columns = 1



class CONTRACTIVE_GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super(CONTRACTIVE_GGCNN2, self).__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True)
        #nn.ReLU(inplace=True),
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True)
        #nn.ReLU(inplace=True),
        self.pool=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        #nn.MaxPool2d(kernel_size=2, stride=2),

        self.conv3 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True)
        #nn.ReLU(inplace=True),
        self.conv4 = nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True)
        #nn.ReLU(inplace=True),
        self.pool1=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        #nn.MaxPool2d(kernel_size=2, stride=2),

        # Dilated convolutions.
        self.conv5 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True)
        #nn.ReLU(inplace=True),
        self.conv6 = nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True)
        #nn.ReLU(inplace=True),

        # Output layers
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], 3, stride=2, padding=1, output_padding=1)
        #nn.ReLU(inplace=True),
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[3], 3, stride=2, padding=1, output_padding=1)
        #nn.ReLU(inplace=True)

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool1(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        output_e = F.relu(self.conv6(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output,output_e

    # define the Contractive loss function
    def loss_function(self,img_in, output_e, pos_pred, cos_pred, sin_pred, width_pred, y_pos, y_cos, y_sin, y_width,lamda = 1e-4):
        h = output_e
        lam = 1e-4
        device = torch.device("cuda:0")

        p_loss = F.mse_loss(pos_pred, y_pos) 
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
        # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
        # opposed to #1
        output_e.backward(torch.ones(output_e.size()).to(device), retain_graph=True)
        loss2 = torch.sqrt(torch.sum(torch.pow(img_in.grad, 2))) # THE CORRECTION
        img_in.grad.data.zero_()
        p_loss = p_loss + (lamda*loss2) 
        cos_loss = cos_loss + (lamda*loss2) 
        sin_loss = sin_loss + (lamda*loss2) 
        width_loss = width_loss + (lamda*loss2) 

        

        return p_loss, cos_loss, sin_loss, width_loss

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc

        xc.retain_grad()
        xc.requires_grad_(True)

        pos_pred, cos_pred, sin_pred, width_pred, output_e = self(xc)

 

        pp = pos_pred[0].permute(1, 2, 0).cpu().detach().numpy() 
        cp = cos_pred[0].permute(1, 2, 0).cpu().detach().numpy() 
        sp = sin_pred[0].permute(1, 2, 0).cpu().detach().numpy() 
        wp = width_pred[0].permute(1, 2, 0).cpu().detach().numpy()


        yp = y_pos[0].permute(1, 2, 0).cpu().detach().numpy() 
        yc = y_cos[0].permute(1, 2, 0).cpu().detach().numpy() 
        ys = y_sin[0].permute(1, 2, 0).cpu().detach().numpy() 
        yw = y_width[0].permute(1, 2, 0).cpu().detach().numpy() 
        

        pp = pp[:, :, 0]
        cp = cp[:, :, 0]
        sp = sp[:, :, 0]
        wp = wp[:, :, 0]
        yp = yp[:, :, 0]
        yc = yc[:, :, 0]
        ys = ys[:, :, 0]
        yw = yw[:, :, 0]

        label_horizontal = np.vstack((yp,yc,ys,yw))
        pred_horizontal = np.vstack((pp,cp,sp,wp))
        total_data = np.hstack((pred_horizontal,label_horizontal))

        p_loss,  cos_loss, sin_loss, width_loss = self.loss_function(xc, output_e, pos_pred, cos_pred, sin_pred, width_pred, y_pos, y_cos, y_sin, y_width)
        xc.requires_grad_(False)
 
        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
