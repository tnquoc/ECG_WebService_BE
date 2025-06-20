import torch
import torch.nn as nn


# model
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        if in_planes == planes and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv1d(in_planes, planes, kernel_size=1,
                                      padding=0, stride=stride, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + shortcut)

        return x


class SelfAttentionModule(nn.Module):
    def __init__(self, channels, r=4):
        super(SelfAttentionModule, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels // r, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels // r, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        f_conv1 = self.conv1(x)
        f_trans = f_conv1.transpose(2, 1)
        f_sa = torch.matmul(f_conv1, f_trans)
        f_conv2 = torch.matmul(f_sa, f_conv1)
        f_output = self.conv2(f_conv2)

        return f_output


class Resnet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Resnet18, self).__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

        self.self_attn = SelfAttentionModule(128, 4)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.mlp_head = nn.Linear(128 * 38, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.self_attn(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp_head(x)

        return x
