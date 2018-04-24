import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding, softmax
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20, use_gpu=False):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.loc_loss = nn.SmoothL1Loss(size_average=False)

    def focal_loss(self, x, y, gamma=2, alpha=None):
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
        N = x.size(0)
        C = x.size(1)
        if alpha is None:
            alpha = 0.25 * Variable(torch.ones(self.num_classes+1, 1)).type(self.FloatTensor)
        ids = y.view(-1, 1)
        alpha = alpha[ids.data.view(-1)]

        class_mask = x.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = y.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)

        p = F.softmax(x, dim=1)  # [N,D]
        pt = torch.sum(p * class_mask, dim=1, keepdim=True)  # [N,]
        log_pt = torch.sum(F.log_softmax(x, dim=1)*class_mask, dim=1, keepdim=True)
        loss = -alpha*(1 - pt).pow(gamma)*log_pt
        loss = loss.squeeze(1)
        loss = torch.mean(loss)
        return loss

    def focal_loss_sigmoid(self, x, y):
        """
        Sigmoid version of focal loss.

        This is described in the original paper.
        With BCELoss, the background should not be counted in num_classes.

        Args:
          x: (tensor) predictions, sized [N,D].
          y: (tensor) targets, sized [N,].

        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

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
        loc_loss = self.loc_loss(masked_loc_preds, masked_loc_targets)

        ################################################################
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_sigmoid(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.5f | cls_loss: %.5f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        if loc_loss.data[0] == 0:
            loss = cls_loss
        else:
            loss = (loc_loss+cls_loss)/num_pos
        return loss


if __name__ == '__main__':
    pass
