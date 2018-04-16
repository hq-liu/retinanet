import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from model.loss import FocalLoss
from model.retina_shuffle import RetinaNet_Shuffle
from model.retina_net import RetinaNet
from data_utils.data_input import ListDataset
from torch.autograd import Variable
from model.convert_model import convert_res50, convert_shuffle_net


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    # net.freeze_bn()
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
        # torch.cuda.empty_cache()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=False,
                        action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu', default=False, help='Use Gpu or not')
    parser.add_argument('--device', default=0, help='Which gpu is using')
    parser.add_argument('--model', default='shufflenet', help='shufflenet or res50')
    parser.add_argument('--epoch', default=200, help='max training epochs')
    parser.add_argument('--batch_size', default=4, help='batch size')
    parser.add_argument('--input_size', default=300, help="input images' size")
    parser.add_argument('--num_classes', default=20, help='Number of classes')
    args = parser.parse_args()

    use_gpu = args.gpu
    if use_gpu:
        torch.cuda.set_device(args.device)
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ListDataset(root=r'D:\VOCdevkit\VOC2007\JPEGImages',
                           list_file='./data/voc07_train.txt',
                           train=True, transform=transform, input_size=args.input_size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

    testset = ListDataset(root=r'D:\VOCdevkit\VOC2007\JPEGImages',
                          list_file='./data/voc07_train.txt',
                          train=False, transform=transform, input_size=args.input_size)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

    # trainset = ListDataset(root=r'E:\st',
    #                        list_file='./data/data2.txt',
    #                        train=True, transform=transform, input_size=args.input_size)
    # trainloader = DataLoader(trainset, batch_size=args.batch_size,
    #                          shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)
    #
    # testset = ListDataset(root=r'E:\st',
    #                       list_file='./data/data2.txt',
    #                       train=False, transform=transform, input_size=args.input_size)
    # testloader = DataLoader(testset, batch_size=args.batch_size,
    #                         shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

    # Model
    if args.model == 'shufflenet':
        convert_shuffle_net(num_classes=args.num_classes)
        net = RetinaNet_Shuffle(num_classes=args.num_classes)
        net.load_state_dict(torch.load('retina_net_shuffle.pth'))
    else:
        convert_res50(num_classes=args.num_classes)
        net = RetinaNet(num_classes=args.num_classes)
        net.load_state_dict(torch.load('retina_net_res50.pth'))
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    if use_gpu:
        net.cuda()

    criterion = FocalLoss(num_classes=args.num_classes, use_gpu=use_gpu)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        # test(epoch)
        torch.save(net.state_dict(), './checkpoint/ckpt.pth')