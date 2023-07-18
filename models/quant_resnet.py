import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

import brevitas.nn as qnn
from brevitas.quant import IntBias
from .common import CommonUintActQuant, CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

__all__ = [
    'ResNet',
    'quantresnet18',
]

def conv3x3(in_planes: int, out_planes: int, weight_bit_width: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> qnn.QuantConv2d:
    return qnn.QuantConv2d(
        in_channels=in_planes, 
        out_channels=out_planes, 
        kernel_size=3, 
        stride=stride,
        padding=dilation, 
        groups=groups, 
        bias=False, 
        dilation=dilation, 
        weight_quant=CommonIntWeightPerChannelQuant,
        weight_bit_width=weight_bit_width)


def conv1x1(in_planes: int, out_planes: int, weight_bit_width: int, stride: int = 1) -> qnn.QuantConv2d:
    return qnn.QuantConv2d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False, 
        weight_quant=CommonIntWeightPerChannelQuant,
        weight_bit_width=weight_bit_width)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            weight_bit_width: int,
            act_bit_width: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            activation_scaling_per_channel = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, weight_bit_width, stride, dilation=dilation)
        self.bn1 = norm_layer(num_features=planes)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            return_quant_tensor=True)
        self.conv2 = conv3x3(planes, planes, weight_bit_width, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = out + identity
        out = self.relu2(out) + self.relu2(identity)
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            weight_bit_width: int,
            act_bit_width: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, weight_bit_width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, weight_bit_width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, weight_bit_width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
        act_quant=CommonUintActQuant,
        bit_width=act_bit_width,
        return_quant_tensor=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu2(out) + self.relu2(identity)
        return out

class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            weight_bit_width: int,
            act_bit_width: int,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 =qnn.QuantConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, weight_bit_width=8)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            return_quant_tensor=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], weight_bit_width, act_bit_width)
        self.layer2 = self._make_layer(block, 128, layers[1], weight_bit_width, act_bit_width, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], weight_bit_width, act_bit_width, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], weight_bit_width, act_bit_width, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = qnn.QuantAvgPool2d(kernel_size=7, bit_width=weight_bit_width)
        # self.fc = qnn.QuantLinear(512 * block.expansion, num_classes, 
        #     bias=True, 
        #     bias_quant=IntBias,
        #     weight_bit_width=weight_bit_width,
        #     weight_quant=CommonIntWeightPerTensorQuant)

        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    weight_bit_width: int, act_bit_width: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, weight_bit_width, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, weight_bit_width, act_bit_width, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, weight_bit_width, act_bit_width, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> List:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x0 = self.maxpool(x)
        
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # out = self.avgpool(x4)
        # out = torch.flatten(out, 1)
        # out = self.fc(out)

        return [x0, x1, x2, x3, x4]
        # return out

    def forward(self, x: Tensor) -> List:
        return self._forward_impl(x)

    def get_dimensions(self):
        x = torch.rand(1, 3, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x0.shape[1], x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]]

    def get_ouput_stride(self):
        if self.replace_stride_with_dilation == [False, False, False]:
            return None
        elif self.replace_stride_with_dilation == [False, True, True]:
            return 8
        elif self.replace_stride_with_dilation == [False, False, True]:
            return 16
        else:
            raise ValueError


def quantresnet18(weight_bit_width, act_bit_width, output_stride: int = None, **kwargs: Any) -> ResNet:
    if output_stride is None:
        replace_stride_with_dilation = None
    elif output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
    elif output_stride == 16:
        replace_stride_with_dilation = [False, False, True]
    else:
        raise ValueError
    return ResNet(BasicBlock, [2, 2, 2, 2], weight_bit_width, act_bit_width, replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)

