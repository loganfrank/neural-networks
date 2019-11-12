import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class two_layer_cnn(nn.Module):
    def __init__(self, num_classes):
        super(two_layer_cnn, self).__init__()
        self.conv = conv3x3(in_planes=3, out_planes=64)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.global_maxpool(x)
        x = self.fc(x)
        return x

class three_layer_cnn(nn.Module):
    def __init__(self, num_classes):
        super(three_layer_cnn, self).__init__()
        self.conv1 = conv3x3(in_planes=3, out_planes=64)
        self.conv2 = conv3x3(in_planes=64, out_planes=128)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.global_maxpool(x)
        x = self.fc(x)
        return x

class one_layer_nn(nn.Module):
    def __init__(self, inputs, num_classes):
        super(one_layer_nn, self).__init__()
        self.fc = nn.Linear(inputs, num_classes, bias=True)
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

class two_layer_nn(nn.Module):
    def __init__(self, inputs, num_classes):
        super(two_layer_nn, self).__init__()
        self.fc1 = nn.Linear(inputs, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
