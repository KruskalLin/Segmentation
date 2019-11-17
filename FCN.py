import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import cv2
from torchviz import make_dot

transform = transforms.Compose([
    transforms.Scale((160, 160), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class TransNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, f1, f2):
        # feats = torch.cat([f1, f2], 1)
        fuse = f1 * f2
        # alpha, beta = torch.chunk(feats, 2, dim=1)
        # fuse = alpha * f2 + beta + f1
        return fuse


class PFCN(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.pretrained_net2 = type(pretrained_net)()
        self.psp = PSPModule(512, 512, (1, 2, 3, 6))
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.relu    = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        self.deconv1_lidar = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1_lidar = nn.BatchNorm2d(512)
        self.deconv2_lidar = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2_lidar = nn.BatchNorm2d(256)
        self.deconv3_lidar = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3_lidar = nn.BatchNorm2d(128)
        self.deconv4_lidar = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4_lidar = nn.BatchNorm2d(64)
        self.deconv5_lidar = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5_lidar = nn.BatchNorm2d(32)
        self.classifier_lidar = nn.Conv2d(32, n_class, kernel_size=1)

        self.linear1 = nn.Conv2d(2, 2, kernel_size=1)
        self.linear2 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x, lidar):
        lidar = F.interpolate(input=lidar, size=(160, 160), mode='bilinear')
        lidar = self.conv(lidar)
        lidar = self.relu(lidar)
        lidar = self.conv1(lidar)
        lidar = self.relu(lidar)
        lidar = self.pretrained_net2(lidar)
        lidar_x5 = lidar['x5']
        lidar_x4 = lidar['x4']
        lidar_x3 = lidar['x3']
        lidar_x2 = lidar['x2']
        lidar_x1 = lidar['x1']
        score = self.bn1_lidar(self.relu(self.deconv1_lidar(lidar_x5)))
        score = score + lidar_x4
        score = self.bn2_lidar(self.relu(self.deconv2_lidar(score)))
        score = score + lidar_x3
        score = self.bn3_lidar(self.relu(self.deconv3_lidar(score)))
        score = score + lidar_x2
        score = self.bn4_lidar(self.relu(self.deconv4_lidar(score)))
        score = score + lidar_x1
        score = self.bn5_lidar(self.relu(self.deconv5_lidar(score)))
        lidar = self.classifier_lidar(score)

        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x5 = self.psp(x5)
        x5 = x5 + lidar_x5
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4 + lidar_x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3 + lidar_x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        mean1 = self.linear1(score)
        mean2 = self.linear2(lidar)
        mean = mean1 + mean2
        return score, lidar, mean


class VGGNet(VGG):
    def __init__(self, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:  
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg-Net config 
# Vgg网络结构配置
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# make layers using Vgg-Net config(cfg)
# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

'''
VGG-16网络参数

Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)

'''


if __name__ == "__main__":
    x1 = torch.randn(4, 3, 160, 160)
    x3 = torch.randn(4, 1, 360, 1200)
    vgg_model = VGGNet(show_params=False)
    fcn_model = PFCN(pretrained_net=vgg_model, n_class=2)
    a, b, c = fcn_model(x1, x3)
    dot = make_dot(c.mean(), params=dict(fcn_model.named_parameters()))
    dot.render('graph.png')
    pass