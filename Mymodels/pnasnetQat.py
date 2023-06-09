'''PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class SepConv(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        return F.relu(y1+y2)

class CellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellB, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        self.relu_left = nn.ReLU(inplace = True)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu_right = nn.ReLU(inplace = True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=self.stride, padding=1)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = self.max_pool(x)
        if self.stride==2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = self.relu_left(nn.quantized.FloatFunctional().add(y1, y2))
        b2 = self.relu_right(nn.quantized.FloatFunctional().add(y3, y4))
        y = nn.quantized.FloatFunctional().cat(x=[b1,b2], dim=1)
        return self.relu2((self.bn2(self.conv2(y))))

class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes):
        super(PNASNet, self).__init__()
        self.quant = QuantStub()
        self.in_planes = num_planes
        self.cell_type = cell_type

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)
        self.relu1 = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes*2)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6)
        self.layer4 = self._downsample(num_planes*4)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6)

        self.avg_pool = nn.AvgPool2d(kernel_size=8)

        self.linear = nn.Linear(num_planes*4, 10)

        self.dequant = DeQuantStub()

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = self.quant(x)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = self.linear(out.view(out.size(0), -1))
        out = self.dequant(out)
        return out
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        for module in self.modules():
            if type(module) == CellB:
                torch.quantization.fuse_modules(module, ['conv2', 'bn2', 'relu2'], inplace=True)
                if module.stride == 2:
                    torch.quantization.fuse_modules(module, ['conv1', 'bn1'], inplace=True)
            if type(module) == SepConv:
                torch.quantization.fuse_modules(module, ['conv1', 'bn1'], inplace=True)
        


def PNASNetA():
    return PNASNet(CellA, num_cells=6, num_planes=44)

def PNASNetB():
    return PNASNet(CellB, num_cells=6, num_planes=32)


def test():
    net = PNASNetB()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
