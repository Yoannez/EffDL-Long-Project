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
from Mymodels import *
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
    transforms.RandomCrop(32, padding=4),
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
print('==> Building PNASNetB model..')

net = PNASNetB()

print("Number of parameters: ", count_parameters(net))
train_acc_plot = []
train_loss_plot = []
test_acc_plot = []
test_loss_plot = []
lr_values_plot = []

net = net.to(device)

results_path = './../results/pnasnet_QAT/'
load_path = './../results/pnasnet_mixup/'
result_name = 'pnasnet_QAT.pth'
final_result_name = 'densenet_cifar200.pth'
load_checkpoint = 'pnasnet_mixup.pth'

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(load_path+load_checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print("Best Acc: ", best_acc)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.eval()
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
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'train loss': train_loss_plot,
        'train acc': train_acc_plot,
    }
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    torch.save(state, results_path+result_name)

# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
#     scheduler.step()

net.eval()
net.fuse_model()
print_size_of_model(net)

net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
print(net.qconfig)

torch.quantization.prepare_qat(net, inplace=True)

for epoch in range(10):
    train(epoch)
    scheduler.step()

net.to('cpu')
torch.quantization.convert(net, inplace=True)
print_size_of_model(net)
test(0)