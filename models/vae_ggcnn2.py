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

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]
fig = plt.figure()
rows = 2
columns = 1

device = torch.device("cuda:0")

class VAE_GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super(VAE_GGCNN2, self).__init__()

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
        self.fc1 = nn.Linear(filter_sizes[2]*75*75, 512)
        self.fc2 = nn.Linear(filter_sizes[2]*75*75, 512)
        self.fc3 = nn.Linear(512, filter_sizes[2]*75*75)

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
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(device)
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool1(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        x = torch.flatten(x, 1)        
        z, mu, logvar = self.bottleneck(x)

        x = self.fc3(z)
        x = x.view(-1,32,75,75)

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output, mu, logvar

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc

        pos_pred, cos_pred, sin_pred, width_pred, mu, logvar = self(xc)

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

        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


        p_loss = F.mse_loss(pos_pred, y_pos) + KLD
        cos_loss = F.mse_loss(cos_pred, y_cos) + KLD
        sin_loss = F.mse_loss(sin_pred, y_sin) + KLD
        width_loss = F.mse_loss(width_pred, y_width) + KLD

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
