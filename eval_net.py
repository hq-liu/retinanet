import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.autograd import Variable
from data_utils.transform import resize
from data_utils.data_input import ListDataset
from data_utils.eval import voc_eval
from model.retina_shuffle import RetinaNet_Shuffle
from model.retina_net import RetinaNet
from data_utils.encoder import DataEncoder

from PIL import Image
from PIL import ImageDraw
import time


class Eval_net():
    def __init__(self, img_size, use_gpu=False):
        self.FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor

        print('Loading model..')
        self.net = RetinaNet_Shuffle(num_classes=4)
        self.net.load_state_dict(torch.load('./checkpoint/ckpt2.pth', map_location={'cuda:2':'cuda:0'}))
        self.net.type(self.FloatTensor)
        self.net.eval()

        print('Preparing dataset..')
        self.img_size = img_size
        self.box_coder = DataEncoder()

        self.pred_boxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.gt_boxes = []
        self.gt_labels = []

        self.gt_difficults=None
        # with open('./data/voc07_test_difficult.txt') as f:
        #     self.gt_difficults = []
        #     for line in f.readlines():
        #         line = line.strip().split()
        #         d = [int(x) for x in line[1:]]
        #         self.gt_difficults.append(d)

    def transform_with_boxes(self, img, boxes, labels):
        img, boxes = resize(img, boxes, size=(self.img_size, self.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, boxes, labels

    def eval(self):
        for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
            print('%d/%d' % (i, len(dataloader)))
            self.gt_boxes.append(box_targets.squeeze(0))
            self.gt_labels.append(label_targets.squeeze(0))

            inputs = Variable(inputs).type(self.FloatTensor)
            loc_preds, cls_preds = self.net(inputs)
            box_preds, label_preds, score_preds = self.box_coder.decode(loc_preds.data.squeeze(),
                                                                        cls_preds.data.squeeze(),
                                                                        input_size=self.img_size)
            self.pred_boxes.append(box_preds)
            self.pred_labels.append(label_preds)
            self.pred_scores.append(score_preds)

            print(voc_eval(
                self.pred_boxes, self.pred_labels, self.pred_scores,
                self.gt_boxes, self.gt_labels, self.gt_difficults,
                iou_thresh=0.5, use_07_metric=True))

    def draw_picture(self):
        print('Loading image..')
        img = Image.open(r'C:\Users\Administrator\Pictures\test\10.jpg')
        w = h = self.img_size
        img = img.resize((w, h))

        print('Predicting..')
        tic = time.time()
        tic = int(round(tic * 1000))
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True).type(self.FloatTensor)
        loc_preds, cls_preds = self.net(x)

        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        toc = time.time()
        toc = int(round(toc * 1000))
        print((toc - tic))
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        img.show()


if __name__ == '__main__':
    eval = Eval_net(300)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ListDataset(root=r'D:\麦当劳',
                          list_file='./data/data.txt',
                          transform=transform, input_size=300, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    eval.eval()
    # eval.draw_picture()