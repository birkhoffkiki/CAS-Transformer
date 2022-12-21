import torch
import torch.nn as nn


class PatchD(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 64, 7, 1, 3, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            # 1/2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #1/4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 1/8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 1/16
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(x)


class ConvD(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 64, 7, 1, 3, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            # 1/2
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            #1/4
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 1/8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 1/16
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        bs = x.shape[0]
        return torch.sigmoid(x.view(bs))


if __name__ == '__main__':
    net = PatchD(8)
    x = torch.randn((12, 8, 128, 128))
    y = net(x)
    gt = torch.zeros_like(y)
    xx = torch.nn.functional.binary_cross_entropy(gt+1, gt)
    print(y)