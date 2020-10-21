import torch
from dataset import YoLoDataset
from util.visualize import drawBox
from util.convert import name
import numpy as np
import cv2

if __name__ == '__main__':
    yolodataset = YoLoDataset(root='../VOCtrainval/VOCdevkit/VOC2012/JPEGImages', txt='../VOC2012/label')
    img, label = yolodataset[0]
    model = torch.load()
    model.eval()
    output = model(img.unsqueeze(0))
    img = img.numpy().transpose((1, 2, 0)) * 0.5 + 0.5  # transforms.Normalize((0.5,), (0.5,))的逆
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i, j, 4] > 0:
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