import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from model.retina_shuffle import RetinaNet_Shuffle
from model.retina_net import RetinaNet
from data_utils.encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

from datetime import datetime
import os
import numpy as np


net = RetinaNet_Shuffle(4)
# net = RetinaNet(20)

net.load_state_dict(torch.load('./checkpoint/ckpt2.pth',
                               map_location={'cuda:0':'cuda:0'}))
net.eval()
target = {'0': "M_Pic:", '1': "M_Word:", '2': "S_Pic:", '3': "S_word:"}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
i = 0
for f in os.listdir(r'E:\test2'):

    img = Image.open(os.path.join(r'E:\test2', f)).convert("RGB")
    w = h = 300
    img1 = img.resize((w, h))

    x = transform(img1)
    x = x.unsqueeze(0)
    x = Variable(x)

    time1 = datetime.now()
    loc_preds, cls_preds = net(x)
    time2 = datetime.now()

    encoder = DataEncoder()
    boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))

    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if score == 0:
            continue
        box[::2] *= img.width/300
        box[1::2] *= img.height/300
        draw.rectangle(list(box), outline='red')
        leftup_corner = list(box)
        leftup_corner = leftup_corner[:2]

        label = str(label.item())
        score_fenshu = str(score)
        score_fenshu = score_fenshu[:4]
        print(label)
        target_name = target[label]
        target_name += score_fenshu
        ft = ImageFont.truetype(r"C:\WINDOWS\Fonts\Arial.TTF", 10)
        draw.text(leftup_corner,target_name, "blue", font=ft)

    use_seconds = (time2-time1).seconds
    use_microseconds = (time2-time1).microseconds
    if use_seconds > 1:
        print("检测用了%.3f秒" % use_seconds)
    else:
        print("检测用了%.3f秒" % (use_microseconds/1000000))
    # img.show()
    print(f)
    img.save(os.path.join(r'E:\test2\test2', str(i)+'.jpg'))
    i += 1
