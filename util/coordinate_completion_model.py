import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net

def define_G(init_type='normal', init_gain=0.02):
    net = CoordinateCompletion()
    return init_weights(net, init_type, init_gain)

class CoordinateCompletion(nn.Module):
    def __init__(self):
        super(CoordinateCompletion, self).__init__()
        self.generator = CoorGenerator(input_nc=2+1, output_nc=2, tanh=True)

    def forward(self, coor_xy, UV_texture_mask):
        complete_coor = self.generator(torch.cat((coor_xy, UV_texture_mask), 1))
        return complete_coor

class CoorGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, batch_norm=True, spectral_norm=False, tanh=True):
        super(CoorGenerator, self).__init__()

        block = GatedConv2dWithActivation
        activation = nn.ELU(inplace=True)

        model = [block(input_nc, ngf, kernel_size=5, stride=1, padding=2,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation),\
                  block(ngf, ngf*2, kernel_size=3, stride=2, padding=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation),\
                  block(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation),\
                  block(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1+1, dilation=2,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1+3, dilation=4,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1+7, dilation=8,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1+7, dilation=8,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, dilation=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, dilation=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  nn.Upsample(scale_factor=2, mode='bilinear'),\
                  block(ngf*4, ngf*2, kernel_size=3, stride=1, padding=1, dilation=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  nn.Upsample(scale_factor=2, mode='bilinear'),\
                  block(ngf*2, ngf, kernel_size=3, stride=1, padding=1, dilation=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  block(ngf, int(ngf/2), kernel_size=3, stride=1, padding=1, dilation=1,
                            batch_norm=batch_norm, spectral_norm=spectral_norm, activation=activation), \
                  nn.Conv2d(int(ngf/2), output_nc, kernel_size=3, stride=1, padding=1, dilation=1)]
        if tanh:
            model += [ nn.Tanh()]

        self.model =  nn.Sequential(*model)

    def forward(self, input):
        out = self.model(input)
        return out

class GatedConv2dWithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, spectral_norm=False, activation=torch.nn.ELU(inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm

        self.pad = nn.ReflectionPad2d(padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        if spectral_norm:
            self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
            self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)
        self.activation = activation
        self.sigmoid = torch.nn.Sigmoid()

        if self.batch_norm:
            self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        padded_in = self.pad(input)
        x = self.conv2d(padded_in)
        mask = self.mask_conv2d(padded_in)
        gated_mask = self.sigmoid(mask)
        if self.batch_norm:
            x = self.batch_norm2d(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x * gated_mask
        return x
