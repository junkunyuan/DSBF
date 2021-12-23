"""
    这个是与 main_dg_cdd.py配套的 network
"""
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock,model_urls,Bottleneck
import torch
import torch.nn as nn
from torchvision.models import alexnet
import torch.nn.utils.weight_norm as weightNorm

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

"""IV"""
class FeatBootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori", use_tanh=True):
        super(FeatBootleneck, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)

        self.type = type
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.use_tanh = use_tanh

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x

class Classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="bn"):
        super(Classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.predict_conv_net = nn.Sequential(
            # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            # Max-pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Convolution
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            # Max-pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.predict_fc_net = nn.Sequential(
            # Fully connected layer
            # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
            # convert matrix with 120 features to a matrix of 84 features (columns)
            # nn.Linear(120, 84),
            # nn.ReLU(),
            # convert matrix with 84 features to a matrix of 10 features (columns)
            # nn.Linear(84, 10),
        )
        self.in_features = 128

    def forward(self, x):
        out = self.predict_conv_net(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.predict_fc_net(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.in_features = 512*block.expansion
        self.pecent = 1/3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
    return model

def lenet(pretrained=True, **kwargs):
    model = LeNet()
    if pretrained:
        print("don't have pretrained lenet")
    return model