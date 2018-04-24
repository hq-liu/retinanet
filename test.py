import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from model.retina_shuffle import RetinaNet_Shuffle
from model.retina_net import RetinaNet
from data_utils.encoder import DataEncoder
from PIL import Image, ImageDraw,ImageFont

from datetime import datetime 
time1 = datetime.now()


net = RetinaNet_Shuffle(4)
# net = RetinaNet(20)

net.load_state_dict(torch.load('./checkpoint/ckpt2.pth',
                               map_location={'cuda:0':'cuda:0'}))
net.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
img = Image.open(r'C:\Users\Administrator\Pictures\Testing\st\IMG_20180318_160607_HDR.jpg')

w = h = 300
img1 = img.resize((w, h))
x = transform(img1)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)
encoder = DataEncoder()
boxes, labels,scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
target = {'0': "M_Pic:", '1': "M_Word:", '2': "S_Pic:", '3': "S_word:"}

draw = ImageDraw.Draw(img)
for box, label, score in zip(boxes, labels, scores):
    box[::2] *= img.width/300
    box[1::2] *= img.height/300
    
    draw.rectangle(list(box), outline='red')
    leftup_corner = list(box)
    leftup_corner = leftup_corner[:2]
    
    label = str(label)
    score_fenshu = str(score)
    score_fenshu = score_fenshu[:4]
    target_name = target[label]
    target_name += score_fenshu
    ft = ImageFont.truetype(r"C:\WINDOWS\Fonts\Arial.TTF", 30)
    draw.text(leftup_corner,target_name, "blue", font=ft)


time2 = datetime.now()
use_seconds = (time2-time1).seconds
use_microseconds = (time2-time1).microseconds
if use_seconds > 1:
    print("检测用了%.3f秒" % use_seconds)
else:
    print("检测用了%.3f秒" % (use_microseconds/1000000))
img.show()
