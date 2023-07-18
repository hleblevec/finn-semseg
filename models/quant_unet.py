import torch.nn as nn
from brevitas.nn import QuantConv1d, QuantConv2d, QuantUpsample, QuantReLU
from .common import *

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, weight_bit_width=8, act_bit_width=8):
        super(UnetBlock, self).__init__()

        self.upsample = QuantUpsample(scale_factor=2)
        self.conv1 = QuantConv2d(in_channels, out_channels, 3, bias=False, padding=1, weight_bit_width=weight_bit_width, weight_quant=CommonIntWeightPerChannelQuant)
        self.conv2 = QuantConv2d(out_channels, out_channels, 3, bias=False, padding=1, weight_bit_width=weight_bit_width, weight_quant=CommonIntWeightPerChannelQuant)
        self.conv3 = QuantConv2d(out_channels, out_channels, 3, bias=False, padding=1, weight_bit_width=weight_bit_width, weight_quant=CommonIntWeightPerChannelQuant)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu1 = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)
        self.relu2 = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)
        self.relu3 = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)


    def forward(self, x, side_input):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        # x = x + side_input
        x = self.relu1(x) + self.relu1(side_input)
        # x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class Unet(nn.Module):
    def __init__(self, encoder,  weight_bit_width=8, act_bit_width=8, num_classes=19):
        super(Unet, self).__init__()

        self.encoder = encoder
        encoder_dims = self.encoder.get_dimensions()
        encoder_dims.reverse()
        encoder_dims.pop(-2)

        self.blocks = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            b = UnetBlock(encoder_dims[i], encoder_dims[i + 1], weight_bit_width, act_bit_width)
            self.blocks.append(b)

        self.upsample = QuantUpsample(scale_factor=4)
        self.conv1 = QuantConv2d(encoder_dims[-1], encoder_dims[-1], 3, padding=1, bias=False, weight_bit_width=weight_bit_width, weight_quant=CommonIntWeightPerChannelQuant)
        self.conv2 = QuantConv2d(encoder_dims[-1], num_classes, 1, bias=False, weight_bit_width=8, weight_quant=CommonIntWeightPerChannelQuant)
        self.bn1 = nn.BatchNorm2d(encoder_dims[-1])
        self.relu1 = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=True)
        self.relu2 = QuantReLU(act_quant=CommonUintActQuant, bit_width=act_bit_width, return_quant_tensor=False)

    def forward(self, x):
        outputs = self.encoder(x)
        outputs.reverse()
        outputs.pop(-2)

        x = outputs[0]
        for i in range(len(outputs) - 1):
            x = self.blocks[i](x, outputs[i + 1])

        x = self.upsample(x)
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x = self.relu2(x)

        return x
