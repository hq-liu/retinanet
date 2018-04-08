'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from model.fpn_shuffle_net import FPN_ShuffleNet
from model.retina_shuffle import RetinaNet_Shuffle


print('Loading pretrained shuffle_net model..')
d = torch.load('./shufflenet.pth.tar', map_location=lambda storage, loc: storage)['state_dict']

print('Loading into FPN50..')
fpn = FPN_ShuffleNet(5)
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
net = RetinaNet_Shuffle(5)
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
torch.save(net.state_dict(), 'retina_net_shuffle.pth')
print('Done!')
