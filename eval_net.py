import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
from data_utils.transform import resize
from data_utils.data_input import ListDataset
from data_utils.eval import voc_eval
from model.retina_shuffle import RetinaNet_Shuffle
from data_utils.encoder import DataEncoder

from PIL import Image


class Eval_net():
    def __init__(self, img_size):
        print('Loading model..')
        self.net = RetinaNet_Shuffle(num_classes=20)
        self.net.load_state_dict(torch.load('./checkpoint/ckpt.pth')['net'])
        # net.cuda()
        self.net.eval()

        print('Preparing dataset..')
        self.img_size = img_size

        dataset = ListDataset(root='D:\VOCdevkit\VOC2007\JPEGImages',
                              list_file='./data/voc07_test.txt',
                              transform=self.transform, input_size=img_size, train=False)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        self.box_coder = DataEncoder()

        self.pred_boxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.gt_boxes = []
        self.gt_labels = []

        with open('./data/voc07_test_difficult.txt') as f:
            self.gt_difficults = []
            for line in f.readlines():
                line = line.strip().split()
                d = [int(x) for x in line[1:]]
                self.gt_difficults.append(d)

    def transform(self, img, boxes, labels):
        img, boxes = resize(img, boxes, size=(self.img_size, self.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, boxes, labels

    def eval(self):
        for i, (inputs, box_targets, label_targets) in enumerate(self.dataloader):
            print('%d/%d' % (i, len(self.dataloader)))
            self.gt_boxes.append(box_targets.squeeze(0))
            self.gt_labels.append(label_targets.squeeze(0))

            loc_preds, cls_preds = self.net(Variable(inputs.cuda(), volatile=True))
            box_preds, label_preds, score_preds = self.box_coder.decode(
                loc_preds.cpu().data.squeeze(),
                F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                score_thresh=0.01, input_size=self.img_size)

            self.pred_boxes.append(box_preds)
            self.pred_labels.append(label_preds)
            self.pred_scores.append(score_preds)

        print(voc_eval(
            self.pred_boxes, self.pred_labels, self.pred_scores,
            self.gt_boxes, self.gt_labels, self.gt_difficults,
            iou_thresh=0.5, use_07_metric=True))

