"""
convert res50 or shuffle_net to retina_net
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from model.fpn_res50 import FPN50
from model.retina_net import RetinaNet
from model.fpn_shuffle_net import FPN_ShuffleNet
from model.retina_shuffle import RetinaNet_Shuffle


def convert_res50(base_name='resnet50-19c8e357.pth', retina_name='retina_net_res50.pth',
                  num_classes=20):
    print('Loading pretrained ResNet50 model..')
    d = torch.load(base_name)

    print('Loading into FPN50..')
    fpn = FPN50()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet..')
    net = RetinaNet(num_classes=num_classes)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

    net.fpn.load_state_dict(dd)
    torch.save(net.state_dict(), retina_name)
    print('Done!')


def convert_shuffle_net(base_name='shufflenet.pth.tar', retina_name='retina_net_shuffle.pth',
                        num_classes=20):
    print('Loading pretrained shuffle_net model..')
    d = torch.load(base_name, map_location=lambda storage, loc: storage)['state_dict']

    print('Loading into FPN_shuffle_net..')
    fpn = FPN_ShuffleNet()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet..')
    net = RetinaNet_Shuffle(num_classes=num_classes)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

    net.fpn.load_state_dict(dd)
    torch.save(net.state_dict(), retina_name)
    print('Done!')


if __name__ == '__main__':
    pass
