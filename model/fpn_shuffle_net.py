"""
Build FPN with Shuffle_Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import shuffle_net
from collections import OrderedDict
import time


class FPN_ShuffleNet(nn.Module):
    def __init__(self, in_channels=3, groups=3):
        super(FPN_ShuffleNet, self).__init__()
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = shuffle_net.conv3x3(self.in_channels,self.stage_out_channels[1], stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [input_size/4] 224->56

        # Bottom-up layers
        # Stage 2
        self.stage2 = self._make_stage(2)  # [input_size/8] 224->28
        # Stage 3
        self.stage3 = self._make_stage(3)  # [input_size/16] 224->14
        # Stage 4
        self.stage4 = self._make_stage(4)  # [input_size/32] 224->7

        self.conv6 = nn.Conv2d(self.stage_out_channels[-1], 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(self.stage_out_channels[-1], 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(self.stage_out_channels[-2], 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(self.stage_out_channels[-3], 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.top_layer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.top_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = shuffle_net.ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = shuffle_net.ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
            )
            modules[name] = module

        return nn.Sequential(modules)

    @staticmethod
    def _upsample_add(x, y):
        """
        Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c2 = self.max_pool(c1)  # [24, 56, 56]
        c3 = self.stage2(c2)  # [240, 28, 28]
        c4 = self.stage3(c3)  # [480, 14, 14]
        c5 = self.stage4(c4)  # [960, 7, 7]
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.lat_layer1(c5)
        p4 = self._upsample_add(p5, self.lat_layer2(c4))
        p4 = self.top_layer1(p4)
        p3 = self._upsample_add(p4, self.lat_layer3(c3))
        p3 = self.top_layer2(p3)
        return p3, p4, p5, p6, p7


def test():
    net = FPN_ShuffleNet()
    fms = net(Variable(torch.randn(1, 3, 300, 300)))
    for fm in fms:
        print(fm.size())


if __name__ == '__main__':
    test()
