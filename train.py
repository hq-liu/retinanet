import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from model.loss import FocalLoss
from model.retina_shuffle import RetinaNet_Shuffle
from data_utils.data_input import ListDataset
from torch.autograd import Variable


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs).cuda() if use_gpu else Variable(inputs)
        loc_targets = Variable(loc_targets).cuda() if use_gpu else Variable(loc_targets)
        cls_targets = Variable(cls_targets).cuda() if use_gpu else Variable(cls_targets)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx+1)))


def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs).cuda() if use_gpu else Variable(inputs)
        loc_targets = Variable(loc_targets).cuda() if use_gpu else Variable(loc_targets)
        cls_targets = Variable(cls_targets).cuda() if use_gpu else Variable(cls_targets)

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


# for epoch in range(start_epoch, start_epoch+200):
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=False,
                        action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    # use_gpu = torch.cuda.is_available()
    use_gpu = False
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ListDataset(root='D:\VOCdevkit\VOC2007\JPEGImages',
                           list_file='./data/voc07_train.txt', train=True, transform=transform, input_size=100)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

    testset = ListDataset(root='D:\VOCdevkit\VOC2007\JPEGImages',
                          list_file='./data/voc07_test.txt', train=False, transform=transform, input_size=100)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

    # Model
    net = RetinaNet_Shuffle(5)
    net.load_state_dict(torch.load('./model/retina_net_shuffle.pth'))
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    if use_gpu:
        net.cuda()

    criterion = FocalLoss(5, use_gpu=use_gpu)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
