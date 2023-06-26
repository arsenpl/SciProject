#FCN model definition
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

# VGG Cell
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TrippleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TrippleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FCN8s(nn.Module):


    def __init__(self,in_channels=3, out_channels=1):
        super(FCN8s,self).__init__()
        self.out_channels = out_channels
        # VGG z 13 warstwami konwolucyjnymi
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = TrippleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = TrippleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = TrippleConv(512, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dodatkowe 3 warstwy konwolucyjne
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU())
        self.score_fr = nn.Conv2d(in_channels=4096, out_channels=self.out_channels,kernel_size=1)

        # Część ekspansji mapy
        self.upscore2 = nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=4, stride=2, padding=1)
        self.score_pool4 = nn.Conv2d(in_channels=512, out_channels=self.out_channels,kernel_size=1)
        self.upscore_pool4 = nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=4, stride=2, padding=1)
        self.score_pool3 = nn.Conv2d(in_channels=256, out_channels=self.out_channels,kernel_size=1)
        self.upscore8 = nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels,kernel_size=16, stride=8,padding=4)

    def forward(self, x):
        #print("input", x.shape)
        x1 = self.conv1(x)
        #print("conv1", x1.shape)
        p1 = self.pool1(x1)
        #print("pool1", p1.shape)
        x2 = self.conv2(p1)
        #print("conv2", x2.shape)
        p2 = self.pool2(x2)
        #print("pool2", p2.shape)
        x3 = self.conv3(p2)
        #print("conv3", x3.shape)
        p3 = self.pool3(x3)
        #print("pool3", p3.shape)
        x4 = self.conv4(p3)
        #print("conv4", x4.shape)
        p4 = self.pool4(x4)
        #print("pool4", p4.shape)
        x5 = self.conv5(p4)
        #print("conv5", x5.shape)
        p5 = self.pool5(x5)
        #print("pool5", p5.shape)
        x6 = self.conv6(p5)
        #print("conv6",x6.shape)
        x7 = self.conv7(x6)
        #print("conv7",x7.shape)
        sf = self.score_fr(x7)
        #print("score_fr", sf.shape)
        u2 = self.upscore2(sf)
        #print("upscore2", u2.shape)
        s4 = self.score_pool4(p4)
        #print("score_pool4", s4.shape)
        s4= TF.resize(s4, u2.shape[2:])
        #print(s4.shape," ", u2.shape)
        f4 = torch.add(s4, u2)
        #print("add score_pool4 & upscore2", f4.shape)
        u4 = self.upscore_pool4(f4)
        #print("12", u4.shape)
        s3 = self.score_pool3(p3)
        s3 = TF.resize(s3, u4.shape[2:])
        #print(s3.shape, " ", u4.shape)
        f3 = torch.add(s3, u4)
        #print("13", f3.shape)
        out = self.upscore8(f3)
        #print("14", out.shape)
        out = TF.resize(out, x.shape[2:])
        return out


def test():
    x = torch.randn((3, 1, 500, 500))
    model = FCN8s(in_channels=1,out_channels=1)

    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()