import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding, softmax
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, use_gpu=False):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

    def focal_loss(self, x, y, gamma=2, alpha=0.25):
        """
        Focal loss.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)\
        where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
        s_j is the unnormalized score for class j.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        """
        t = one_hot_embedding(y.data.cpu(), self.num_classes+1)
        t = t[1:, 1:]
        t = Variable(t).type(self.FloatTensor)  # [N,D]
        p = F.softmax(x, dim=1)  # [N,D]
        pt = (p * t).sum(1)  # [N,]
        loss = -alpha*(1-pt).pow(gamma)*torch.log(pt)
        return loss.mean()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """
        Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss)/num_pos
        return loss


if __name__ == '__main__':
    f = FocalLoss(3)
    x = torch.rand(3, 3)
    y = torch.zeros(4,).long()
    y[0] = 0
    y[1] = 1
    y[2] = 2
    y[3] = 3
    x, y = Variable(x), Variable(y)
    z1 = f.focal_loss(x, y)
    print(z1)
