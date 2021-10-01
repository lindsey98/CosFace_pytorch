from __future__ import print_function
from __future__ import division
import argparse
import os
import time

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import dataset
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
import lfw_eval
import net
# from dataset import ImageList
# import lfw_eval
import layer

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not  (default: False)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='checkpoint/',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--nb_workers', type=int, default=4,
                    help='how many workers to load data')

# Network
parser.add_argument('--dataset', type=str, default='logo2k_super100', help='Which Database')
parser.add_argument('--train_root', type=str, default='/home/ruofan/PycharmProjects/dml_cross_entropy/data/logo2k_super100/train', help='training data path')
parser.add_argument('--test_root', type=str, default='/home/ruofan/PycharmProjects/dml_cross_entropy/data/logo2k_super100/test', help='training data path')
parser.add_argument('--network', type=str, default='resnet50', help='Which network for train. (sphere20, sphere64, LResNet50E_IR, resnet50)')
# Classifier
parser.add_argument('--num_class', type=int, default=107, help='number of people(class)')
parser.add_argument('--IPC', type=int, default=8, help='number of instances per class in a batch')
parser.add_argument('--classifier_type', type=str, default='MCP', help='Which classifier for train. (MCP, AL, L)')
# LR policy
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# if args.database is 'WebFace':
#     args.train_list = '/home/wangyf/dataset/CASIA-WebFace/CASIA-WebFace-112X96.txt'
#     args.num_class = 10572
#     args.step_size = [16000, 24000]
# elif args.database is 'VggFace2':
#     args.train_list = '/home/wangyf/dataset/VGG-Face2/VGG-Face2-112X96.txt'
#     args.num_class = 8069
#     args.step_size = [80000, 120000, 140000]
# else:
#     raise ValueError("NOT SUPPORT DATABASE! ")


def main():
    # --------------------------------------model----------------------------------------
    if args.network is 'sphere20':
        model = net.sphere(type=20, is_gray=args.is_gray)
    elif args.network is 'sphere64':
        model = net.sphere(type=64, is_gray=args.is_gray)
    elif args.network is 'LResNet50E_IR':
        model = net.LResNet50E_IR(is_gray=args.is_gray)
    elif args.network is 'resnet50':
        model = net.resnet50()
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

    model = torch.nn.DataParallel(model).to(device)
    # print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model.module.save(args.save_path + 'CosFace_0_checkpoint.pth')

    classifier = {
        'MCP': layer.MarginCosineProduct(2048, args.num_class).to(device),
        'AL' : layer.AngleLinear(2048, args.num_class).to(device),
        'L'  : torch.nn.Linear(2048, args.num_class, bias=False).to(device)
    }[args.classifier_type]

    trn_dataset = dataset.load(
            name = args.dataset,
            root = args.train_root,
            mode = 'train',
            transform = dataset.utils.make_transform(
                is_train = True,
                is_inception = False
            ))
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.batch_size, images_per_class = args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size = args.batch_size, drop_last = True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers = args.nb_workers,
        pin_memory = True,
        batch_sampler = batch_sampler
    )

    ev_dataset = dataset.load(
        name=args.dataset,
        root=args.test_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=False
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    print(len(dl_tr.dataset.nb_classes()))
    print(len(dl_tr.dataset))
    print(len(dl_ev.dataset))
    # print('Number of Identities: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # ----------------------------------------train----------------------------------------
    # lfw_eval.eval(args.save_path + 'CosFace_0_checkpoint.pth')
    for epoch in range(1, args.epochs + 1):
        train(dl_tr, model, classifier, criterion, optimizer, epoch)
        model.module.save(args.save_path + args.dataset + '_CosFace_' + str(epoch) + '_checkpoint.pth')
        lfw_eval.evaluate(model, dl_ev)

    print('Finished Training')
    lfw_eval.evaluate(model, dl_ev)


def train(dl_tr, model, classifier, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(dl_tr, 1):
        iteration = (epoch - 1) * len(dl_tr) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)
        data, target = data.to(device), target.to(device)
        # compute output
        output = model(data)
        if isinstance(classifier, torch.nn.Linear):
            output = classifier(output)
        else:
            output = classifier(output, target)
        loss = criterion(output, target)
        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            if args.classifier_type is 'MCP':
                INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(classifier.m, classifier.s)
            elif args.classifier_type is 'AL':
                INFO = ' lambda: {:.4f}'.format(classifier.lamb)
            else:
                INFO = ''
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(dl_tr.dataset), 100. * batch_idx / len(dl_tr),
                    iteration, loss_display, time_used, args.log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


if __name__ == '__main__':
    print(args)
    main()
