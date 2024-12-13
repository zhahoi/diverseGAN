import torch
from torch import nn
from model.layernorm import LayerNorm

Norm = LayerNorm

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        
        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


class Conv2DLReLU(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=0, negative_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size, stride, padding)
        self.ln = Norm(outc)
        self.llr = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.llr(self.ln(self.conv(x)))


class Conv2DInstLReLU(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=0, negative_slope=0.2, is_inst=True):
        super().__init__()
        self.is_inst = is_inst
        self.conv = nn.Conv2d(inc, outc, kernel_size, stride, padding)
        self.inst = nn.InstanceNorm2d(outc, affine=True)
        self.llr = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        if self.is_inst:
            return self.llr(self.inst(self.conv(x)))
        else:
            return self.llr(self.conv(x))


class Conv2DTransposeLReLU(nn.Module):
    def __init__(self, inc, outc, bilinear=True):
        super().__init__()
        if bilinear:
            self.deconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.deconv = nn.ConvTranspose2d(inc, outc, kernel_size=2, stride=2, padding=0)
        self.ln = Norm(outc)
        self.llr = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.llr(self.ln(self.deconv(x)))


class SwishMod(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, 1, 1)
        self.ln = Norm(outc)

    def forward(self, x):
        _x = torch.sigmoid(self.ln(self.conv(x)))
        return x.mul(_x)


class SwishGatedBlock(nn.Module):
    def __init__(self, inc, outc, cat=False, conv1x1=True, dropout=False):
        super().__init__()
        self.conv1x1 = conv1x1

        if conv1x1:
            self.conv0 = Conv2DLReLU(inc, outc, padding=1)
            inc = outc
            self.conv1 = Conv2DLReLU(inc, outc, padding=1)
        else:
            self.conv1 = Conv2DLReLU(inc, outc, padding=1)
        self.conv2 = Conv2DLReLU(outc, outc, padding=1)

        self.pooling = nn.MaxPool2d(2)
        if cat:
            self.deconv1 = Conv2DTransposeLReLU(outc, outc)
            self.deconv2 = Conv2DTransposeLReLU(inc, outc)
            self.swish_mod = SwishMod(outc, outc)
        else:
            self.swish_mod = SwishMod(inc, inc)

    def forward(self, inputs, cat=None):
        if self.conv1x1:
            inputs = self.conv0(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)

        if cat is None:
            # downsampling
            sgb_op = self.pooling(x)
            swish = self.pooling(inputs)
            swish = self.swish_mod(swish)
            concat = [sgb_op, swish]
        else:
            sgb_op = self.deconv1(x)
            swish = self.deconv2(inputs)
            swish = self.swish_mod(swish)
            concat = [sgb_op, swish, cat]

        return torch.cat(concat, dim=1), x