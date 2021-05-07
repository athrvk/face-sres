import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels, 0.8)
        )

    def forward(self, x):
        return x + self.net(x)
        
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, scaleFactor, k=3, p=1):
        super(UpsampleBLock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scaleFactor ** 2), kernel_size=k, padding=p),
            nn.BatchNorm2d(in_channels * (scaleFactor ** 2)),
            nn.PixelShuffle(scaleFactor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)
        
class Generator(nn.Module):
    def __init__(self, z_dim=3, im_chan=3, hidden_dim=64, n_residual=8*2):
        super(Generator, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        for i in range(n_residual):
            self.add_module('residual' + str(i+1), ResidualBlock(hidden_dim, hidden_dim))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim, 0.8),
            nn.PReLU()
        )
        
        self.upsample = nn.Sequential(
        	UpsampleBLock(hidden_dim, 2),
        	UpsampleBLock(hidden_dim, 2),
        	nn.Conv2d(hidden_dim, im_chan, kernel_size=9, padding=4)
        )

    def forward(self, x):
        #print ('Generator input size :' + str(x.size()))
        y = self.conv1(x)
        cache = y.clone()
        
        for i in range(self.n_residual):
            y = self.__getattr__('residual' + str(i+1))(y)
            
        y = self.conv2(y)
        y = self.upsample(y + cache)
        #print ('Generator output size :' + str(y.size()))
        return (torch.sigmoid(y))



