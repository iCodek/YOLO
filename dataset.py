import torch
import os
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from util.convert import name
from util.visualize import drawBox


class YoLoDataset(Dataset):
    def __init__(self, root, txt, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        self.size = 448
        self.grid_num = 7
        self.transform = transform
        self.labels = []
        self.boxes = []
        txts = os.listdir(txt)
        self.txts = [os.path.join(txt, t) for t in txts]
        self.imgs = [os.path.join(root, t.replace('txt', 'jpg')) for t in txts]
        for t in self.txts:
            label = open(t).read().strip().split()
            label = [float(x) for x in label]
            boxes = []
            labels = []
            for k in range(len(label) // 5):
                s = k * 5
                c = int(label[s])
                x = label[s + 1]
                y = label[s + 2]
                w = label[s + 3]
                h = label[s + 4]
                boxes.append([x, y, w, h])
                labels.append(c + 1)
            self.boxes.append(torch.Tensor(boxes))
            self.labels.append(torch.LongTensor(labels))

    def __len__(self):
        return len(self.txts)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = self.transform(img)
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        target = self.encoder(boxes, labels)
        return img, target

    def encoder(self, boxes, labels):
        """
        boxes (tensor) [[x,y,w,h],[]]
        labels (tensor) [...]
        return 7x7x30
        """
        grid_num = self.grid_num
        target = torch.zeros((grid_num, grid_num, 30))
        cell_size = 1. / grid_num
        for i in range(boxes.size()[0]):
            cxcy_sample = boxes[i][:2]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = boxes[i][2:]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = boxes[i][2:]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target


if __name__ == '__main__':
    yolodataset = YoLoDataset(root='VOCtrainval/VOCdevkit/VOC2012/JPEGImages', txt='VOC2012/label')
    img, label = yolodataset[0]
    img = img.numpy().transpose((1, 2, 0)) * 0.5 + 0.5  # transforms.Normalize((0.5,), (0.5,))的逆
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j, 10:].sum() > 0:
                xy = ((label[i, j, :2] + torch.tensor(
                    [j, i])) * yolodataset.size / yolodataset.grid_num).numpy().astype(np.int32)
                wh = (label[i, j, 2:4] * yolodataset.size / 2).numpy().astype(np.int32)
                c = int(label[i, j, 4])
                if c != 0:
                    print(xy, name[torch.argmax(label[i, j, 10:])])
                    drawBox(img, tuple(xy - wh), tuple(xy + wh), label=name[torch.argmax(label[i, j, 10:])])
                # xy = ((label[i, i, 5:7] + i) * yolodataset.size / yolodataset.grid_num).numpy().astype(np.int32)
                # wh = ((label[i, i, 7:9] + i) * yolodataset.size / 2 / yolodataset.grid_num).numpy().astype(np.int32)
                # c = int(label[i, i, 9])
                # if c != 0:
                #     print(xy, name[torch.argmax(label[i, i, 10:])])
                #     drawBox(img, tuple(xy - wh), tuple(xy + wh), label=name[torch.argmax(label[i, i, 10:])])
    cv2.imshow('img', img)
    cv2.waitKey()
