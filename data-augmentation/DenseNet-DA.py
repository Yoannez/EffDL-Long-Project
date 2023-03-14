import sys
sys.path.append("..")
from torchvision.datasets import CIFAR10
import numpy as np 
import torchvision.transforms as transforms
import torch 
from torch.utils.data.dataloader import DataLoader
'''Train CIFAR10 with PyTorch.'''
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
from models import *
from utils import  * 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

### Data
print('==> Preparing data..')
## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 will be downloaded in the following folder
rootdir = './../data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32) 
print(f"CIFAR10 dataset has {len(c10train)} samples")

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# Model
print('==> Building DenseNet121 model..')

net = DenseNet121()

print("Number of parameters: ", count_parameters(net))
train_acc_plot = []
train_loss_plot = []
test_acc_plot = []
test_loss_plot = []
lr_values_plot = []

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('results'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./results/random-crop/random-crop.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    train_acc_plot = checkpoint['train acc']
    train_loss_plot = checkpoint['train loss']
    test_acc_plot = checkpoint['test acc']
    test_loss_plot = checkpoint['test loss']
    lr_values_plot = checkpoint['lr values']
    print("Best Acc: ", best_acc)
    plot_loss(train_loss_plot, test_loss_plot, 'loss.jpg')
    plot_acc(train_acc_plot, test_acc_plot, 'acc.jpg')
    plot_lr(lr_values_plot, 'lr.jpg')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc_plot.append(100.*correct/total)
    train_loss_plot.append([train_loss/(batch_idx+1)])


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc_plot.append(100.*correct/total)
    test_loss_plot.append(test_loss/(batch_idx+1))
    lr_values_plot.append(scheduler.get_last_lr())

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'train loss': train_loss_plot,
            'train acc': train_acc_plot,
            'test loss': test_loss_plot,
            'test acc': test_acc_plot,
            'lr values': lr_values_plot
        }
        if not os.path.isdir('results/random-flip'):
            os.mkdir('results/random-flip')
        torch.save(state, './results/random-flip/random-flip.pth')
        best_acc = acc
    
    if epoch == 199:
        print('Finish..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'train loss': train_loss_plot,
            'train acc': train_acc_plot,
            'test loss': test_loss_plot,
            'test acc': test_acc_plot,
            'lr values': lr_values_plot
        }
        if not os.path.isdir('results/random-flip'):
            os.mkdir('results/random-flip')
        torch.save(state, './results/random-flip/random-flip200.pth')


for epoch in range(start_epoch, 200):
    train(epoch)
    test(epoch)
    scheduler.step()


