from mimetypes import init
from utils import var
import torch
from torch import nn
from model.networks import SwishGatedBlock, SwishMod, Conv2DInstLReLU, Conv2DLReLU, ResBlock
from torchsummary import summary

import torch.nn.init as init
from torch.nn.utils.spectral_norm import spectral_norm

class Latent_Discriminator(nn.Module):
    def __init__(self, z_dim=8, n_filters=64, negative_slope=0.2):
        super(Latent_Discriminator, self).__init__()
        # Discriminator with latent space z # (N, 8)
        self.dz = nn.Sequential(
            nn.Linear(z_dim, n_filters, bias=True),
            nn.LayerNorm(n_filters),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(n_filters, n_filters, bias=True),
            nn.LayerNorm(n_filters),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(n_filters, n_filters, bias=True),
            nn.LayerNorm(n_filters),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(n_filters, n_filters, bias=True),
            nn.LayerNorm(n_filters),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Linear(n_filters, 1, bias=True),
            nn.Sigmoid()
        )
        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        out = self.dz(x)
        return out 


class Encoder(nn.Module):
    def __init__(self, in_nc=3, dim=64, z_dim=8):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # n, 3, 128, 128 -> n, 256, 1, 1
        self.encode = nn.Sequential(
            nn.Conv2d(in_nc, dim // 2, kernel_size=7, stride=2, padding=3),
            ResBlock(dim // 2, dim),
            ResBlock(dim, dim * 2),
            ResBlock(dim * 2, dim * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        # n, 256, 1, 1 -> n, 256 -> n, 8
        self.fc_mu = nn.Linear(dim * 4, z_dim)
        # n, 512, 1, 1 -> n, 512 -> n, 8
        self.fc_logvar = nn.Linear(dim * 4, z_dim)

        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
        
    def forward(self, x):
        # n, 3, 128, 128 -> n, 256, 1, 1
        encode = self.encode(x)
        # n, 256, 1, 1 -> n, 256
        encode = torch.flatten(encode, start_dim=1)
        # get mu n, 256 -> n, 8
        mu = self.fc_mu(encode)
        # get logvar n, 256 -> n, 8
        log_var = self.fc_logvar(encode)
        # use reparameter trick
        encoded_z = self.reparemeterize(mu, log_var, self.z_dim)
        return (encoded_z, mu, log_var)
        
    # define reparameter tricks for latent space z
    def reparemeterize(self, mu, log_variance, z_dim):
        std = torch.exp(log_variance / 2)
        random_z = var(torch.randn(1, z_dim))
        encoded_z = (random_z * std) + mu
        return encoded_z


class Generator(nn.Module):
    def __init__(self, z_dim=8):
        super(Generator, self).__init__()

        down_in_channels = [3 + z_dim, 75, 256, 384, 512, 640]
        down_out_channels = [64, 128, 192, 256, 320, 384]
        up_in_channels = [768, 1152, 960, 768, 576, 384]
        up_out_channels = [384, 320, 256, 192, 128, 64]

        self.down0 = SwishGatedBlock(down_in_channels[0], down_out_channels[0], conv1x1=False)
        self.down1 = SwishGatedBlock(down_in_channels[1], down_out_channels[1])
        self.down2 = SwishGatedBlock(down_in_channels[2], down_out_channels[2])
        self.down3 = SwishGatedBlock(down_in_channels[3], down_out_channels[3])
        self.down4 = SwishGatedBlock(down_in_channels[4], down_out_channels[4])
        self.down5 = SwishGatedBlock(down_in_channels[5], down_out_channels[5])

        self.swishmod0 = SwishMod(down_out_channels[0], down_out_channels[0])
        self.swishmod1 = SwishMod(down_out_channels[1], down_out_channels[1])
        self.swishmod2 = SwishMod(down_out_channels[2], down_out_channels[2])
        self.swishmod3 = SwishMod(down_out_channels[3], down_out_channels[3])
        self.swishmod4 = SwishMod(down_out_channels[4], down_out_channels[4])
        self.swishmod5 = SwishMod(down_out_channels[5], down_out_channels[5])

        self.up0 = SwishGatedBlock(up_in_channels[0], up_out_channels[0], cat=True)
        self.up1 = SwishGatedBlock(up_in_channels[1], up_out_channels[1], cat=True)
        self.up2 = SwishGatedBlock(up_in_channels[2], up_out_channels[2], cat=True)
        self.up3 = SwishGatedBlock(up_in_channels[3], up_out_channels[3], cat=True)
        self.up4 = SwishGatedBlock(up_in_channels[4], up_out_channels[4], cat=True)
        self.up5 = SwishGatedBlock(up_in_channels[5], up_out_channels[5], cat=True)

        self.out = nn.Sequential(
            Conv2DLReLU(down_out_channels[0] * 3, down_out_channels[0], kernel_size=1),
            Conv2DLReLU(down_out_channels[0], down_out_channels[0], kernel_size=3, padding=1),
            Conv2DLReLU(down_out_channels[0], down_out_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(down_out_channels[0], 3, kernel_size=1),
            nn.Tanh()
        )
        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
        
    def forward(self, x, z):
        # print('Encoder')
        # z : (N, z_dim) -> (N, z_dim, 1, 1) -> (N, z_dim, H, W)
        # x_with_z : (N, 3 + z_dim, H, W)
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z], dim=1)

        # [B, 1, 128, 128] -> [B, 65, 64, 64] + [2, 64, 128, 128]
        inputs, conv0 = self.down0(x_with_z)
        # [B, 65, 64, 64] -> [B, 256, 32, 32] + [2, 128, 64, 64]
        inputs, conv1 = self.down1(inputs) 
        # [B, 256, 32, 32] -> [B, 384, 16, 16] + [2, 192, 32, 32]
        inputs, conv2 = self.down2(inputs)
        # [B, 384, 16, 16] -> [B, 512, 8, 8] + [2, 256, 16, 16]
        inputs, conv3 = self.down3(inputs)
        # [B, 512, 8, 8] -> [B, 640, 4, 4] + [2, 320, 8, 8]
        inputs, conv4 = self.down4(inputs)
        # [B, 640, 4, 4] -> [B, 768, 2, 2] + [2, 384, 4, 4]
        inputs, conv5 = self.down5(inputs)

        # print('SwishMod')
        # [2, 64, 128, 128]
        conv0 = self.swishmod0(conv0)
        # [2, 128, 64, 64]
        conv1 = self.swishmod1(conv1)
        # [2, 192, 32, 32]
        conv2 = self.swishmod2(conv2)
        # [2, 256, 16, 16]
        conv3 = self.swishmod3(conv3)
        # [2, 320, 8, 8]
        conv4 = self.swishmod4(conv4)
        # [2, 384, 4, 4]
        conv5 = self.swishmod5(conv5)

        # print('Decoder')
        # [B, 768, 2, 2] -> [B, 1152, 4, 4]
        inputs, _ = self.up0(inputs, cat=conv5)
        # [B, 1152, 4, 4] -> [B, 960, 8, 8]
        inputs, _ = self.up1(inputs, cat=conv4)
        # [B, 960, 8, 8] -> [B, 768, 16, 16]
        inputs, _ = self.up2(inputs, cat=conv3)
        # [B, 768, 16, 16] -> [B, 576, 32, 32]
        inputs, _ = self.up3(inputs, cat=conv2)
        # [B, 576, 32, 32] -> [B, 384, 64, 64]
        inputs, _ = self.up4(inputs, cat=conv1)
        # [B, 384, 64, 64] -> [B, 192, 128, 128]
        inputs, _ = self.up5(inputs, cat=conv0)
        # [B, 192, 128, 128] -> [B, 3, 128, 128]
        out = self.out(inputs)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_filters=64):
        super(Discriminator, self).__init__()   

        self.out = nn.Sequential(
            Conv2DInstLReLU(inc=6, outc=n_filters, kernel_size=4, stride=2, padding=1, is_inst=False),
            Conv2DInstLReLU(inc=n_filters, outc=n_filters * 2, kernel_size=4, stride=2, padding=1),
            Conv2DInstLReLU(inc=n_filters * 2, outc=n_filters * 4, kernel_size=4, stride=2, padding=1),
            Conv2DInstLReLU(inc=n_filters * 4, outc=n_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=0),
        )
        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        out = self.out(x)
        return out


if __name__ == '__main__':
    model = Latent_Discriminator(n_filters=64)
    summary(model=model, input_size=(1, 8))
