#segNet model definition
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# VGG Cell
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, BN_momentum):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TrippleConv(nn.Module):
    def __init__(self, in_channels, out_channels, BN_momentum):
        super(TrippleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BN_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, BN_momentum=0.5):
        super(SegNet, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels

        self.enpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.econv1 = DoubleConv(in_channels, 64, BN_momentum)
        self.econv2 = DoubleConv(64, 128, BN_momentum)
        self.econv3 = TrippleConv(128, 256, BN_momentum)
        self.econv4 = TrippleConv(256, 512, BN_momentum)
        self.econv5 = TrippleConv(512, 512, BN_momentum)

        self.dpool = nn.MaxUnpool2d(2, stride=2)

        self.dconv1 = TrippleConv(512, 512, BN_momentum)
        self.dconv2 = TrippleConv(512, 256, BN_momentum)
        self.dconv3 = TrippleConv(256, 128, BN_momentum)
        self.dconv4 = DoubleConv(128, 64, BN_momentum)
        self.dconv5 = DoubleConv(64, out_channels, BN_momentum)

        self.dense = nn.Linear(out_channels,out_channels)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        #print("input", x.shape)
        x1 = self.econv1(x)
        mp1,i1 = self.enpool(x1)
        sz1 = mp1.size()
        #print("conv1 mp1", mp1.shape)
        x2 = self.econv2(mp1)
        mp2, i2 = self.enpool(x2)
        sz2 = mp2.size()
        #print("conv2 mp2", mp2.shape)
        x3 = self.econv3(mp2)
        mp3, i3 = self.enpool(x3)
        sz3 = mp3.size()
        #print("conv3 mp3", mp3.shape)
        x4 = self.econv4(mp3)
        mp4, i4 = self.enpool(x4)
        sz4 = mp4.size()
        #print("conv4 mp4", mp4.shape)
        x5 = self.econv5(mp4)
        mp5, i5 = self.enpool(x5)
        sz5 = mp5.size()
        #print("conv5 mp5", mp5.shape)
        #print(sz5)
        dp1 = self.dpool(mp5, i5, output_size=sz4)
        x6 =self.dconv1(dp1)
        #print("deconv1 up1", x6.shape)
        dp2 = self.dpool(x6, i4, output_size=sz3)
        x7 = self.dconv2(dp2)
        #print("deconv2 up2", x7.shape)
        dp3 = self.dpool(x7, i3, output_size=sz2)
        x8 = self.dconv3(dp3)
        #print("deconv3 up3", x8.shape)
        dp4 = self.dpool(x8, i2, output_size=sz1)
        x9 = self.dconv4(dp4)
        #print("deconv4 up4", x9.shape)
        dp5 = self.dpool(x9, i1)
        out = self.dconv5(dp5)
        #print("deconv5 up5", x10.shape)
        #dl = self.dense(x10)
        #out = self.soft(out)
        #print("out", out.shape)
        return out
        '''
        ################################################
        self.in_chn = in_channels
        self.out_chn = out_channels

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()
        #print("Stage1: ",x[1:])
        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()
        #print("Stage2: ", x[1:])
        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()
        #print("Stage3: ", x[1:])
        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()
        #print("Stage4: ", x[1:])
        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        #print("Stage5: ", x[1:])
        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        #print("UP Stage1: ", x[1:])
        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))
        #print("Stage2: ", x[1:])
        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))
        #print("Stage3: ", x[1:])
        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))
        #print("Stage4: ", x[1:])
        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)
        #print("Stage5: ", x[1:])
        #x = F.softmax(x, dim=1)

        return x
        ##################################################
        '''
def test():
    x = torch.randn((3, 1, 32, 32))
    #print("BEFORE:",x[1:])
    model = SegNet(in_channels=1,out_channels=1)

    preds = model(x)
    #print("AFTER:",preds[1:])
    #print(x.shape)
    #print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()