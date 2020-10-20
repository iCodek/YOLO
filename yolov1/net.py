import torch.nn as nn


class Yolov1(nn.Module):
    def __init__(self, classNum=20, gridNum=7):
        super(Yolov1, self).__init__()
        print("------Initiating YOLO v1------")
        print('classNum={}, S={}'.format(classNum, gridNum))
        self.c = classNum
        self.g = gridNum
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7//2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1//2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer2 = nn.Sequential(nn.Linear(in_features=4096, out_features=gridNum * gridNum * (2 * 5 + classNum)), nn.Sigmoid())

    def forward(self, input):
        conv_layer1 = self.conv_layer1(input)
        conv_layer2 = self.conv_layer2(conv_layer1)
        conv_layer3 = self.conv_layer3(conv_layer2)
        conv_layer4 = self.conv_layer4(conv_layer3)
        conv_layer5 = self.conv_layer5(conv_layer4)
        conv_layer6 = self.conv_layer6(conv_layer5)
        flatten = conv_layer6.view(conv_layer6.size(0), -1)
        conn_layer1 = self.conn_layer1(flatten)
        conn_layer2 = self.conn_layer2(conn_layer1)
        output = conn_layer2.reshape(conn_layer2.size(0), self.g, self.g, self.c+10)
        return output


if __name__ == '__main__':
    from dataset import YoLoDataset
    from torch.utils.data import DataLoader
    from yolov1.loss import YoloLoss
    yolodataset = YoLoDataset(root='../VOCtrainval/VOCdevkit/VOC2012/JPEGImages', txt='../VOC2012/label')
    trainLoader = DataLoader(yolodataset, batch_size=16, shuffle=True)
    yoloNet = Yolov1()
    trainIter = iter(trainLoader)
    img, target = next(trainIter)
    output = yoloNet(img)
    loss = YoloLoss()
    print(target.size())
    print(output.size())
    print(loss(output, target))
