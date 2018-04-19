import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from model.retina_shuffle import RetinaNet_Shuffle
from model.retina_net import RetinaNet
from data_utils.encoder import DataEncoder
from PIL import Image, ImageDraw


print('Loading model..')
net = RetinaNet(5)
net.load_state_dict(torch.load('./checkpoint/ckpt_res.pth', map_location=lambda storage, loc: storage))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

print('Loading image..')
img = Image.open(r'D:\麦当劳\2018030712(18).jpg')
w = h = 300
img = img.resize((w, h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x)
cls_preds = cls_preds.data.squeeze()
cls_preds = Variable(cls_preds)

print('Decoding..')
encoder = DataEncoder()
boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
