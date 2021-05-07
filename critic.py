import torch
import torch.nn as nn


class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels of the output image, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim), #1st
            self.make_crit_block(hidden_dim, hidden_dim, stride=2), #2nd
            self.make_crit_block(hidden_dim, hidden_dim*2), #3rd
            self.make_crit_block(hidden_dim*2, hidden_dim*2, stride=2), #4th
            self.make_crit_block(hidden_dim*2, hidden_dim*4), #5th (128, 256)
            self.make_crit_block(hidden_dim*4, hidden_dim*4, stride=2), #6th
            self.make_crit_block(hidden_dim*4, hidden_dim*8), #7th
            self.make_crit_block(hidden_dim*8, hidden_dim*8, stride=2), #8th
            nn.AdaptiveAvgPool2d(1), #9th
            self.make_crit_block(hidden_dim*8, hidden_dim*16, kernel_size=1), #10th
            self.make_crit_block(hidden_dim*16, 1, kernel_size=1, final_layer=True)
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block;
        a convolution and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                # nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        # print ('Critic input size :' +  str(image.size()))
        crit_pred = self.crit(image)
        # print ('Critic output size :' +  str(crit_pred.size()))
        return crit_pred.view(len(crit_pred), -1)