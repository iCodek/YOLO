import cv2
import os
import random
import math
import numpy as np
from util.convert import name


def view(img, label_path, num=9, classes=None):
    classes = classes or name
    if os.path.isdir(img):
        imgs = os.listdir(label_path)
        imgs = random.sample(imgs, min(len(imgs), num))
        imgs = [os.path.join(img, x.replace('.txt', '.jpg')) for x in imgs]
        cols = math.ceil(num ** 0.5)
        rows = cols if num > cols * (cols-1) else cols-1
    else:
        imgs = [img]
        rows = cols = 1
    row = [None] * rows
    for i in range(rows * cols):
        if i < len(imgs):
            image = cv2.imread(imgs[i])
            print(imgs[i])
            image = cv2.resize(image, (1200 // cols, 900 // cols))
            label = open(os.path.join(label_path, os.path.basename(imgs[i]).replace('.jpg', '.txt'))).read().strip().split()
            label = [float(x) for x in label]
            for k in range(len(label) // 5):
                s = k * 5
                cls = int(label[s])
                p1, p2 = (label[s + 1] - label[s + 3] / 2, label[s + 2] - label[s + 4] / 2),\
                         (label[s + 1] + label[s + 3] / 2, label[s + 2] + label[s + 4] / 2)
                p1, p2 = (int(p1[0]*image.shape[1]), int(p1[1]*image.shape[0])),\
                         (int(p2[0]*image.shape[1]), int(p2[1]*image.shape[0]))
                drawBox(image, p1, p2, label=classes[cls])
        else:
            image = np.zeros((900 // cols, 1200 // cols, 3), dtype=np.uint8)
        # img = np.vstack((img, img2))  # vstack按垂直方向，hstack按水平方向
        # img = np.concatenate((img, img2), axis=0)  axis=0 按垂直方向，axis=1 按水平方向
        row[i // cols] = np.concatenate((row[i // cols], image), axis=1) if i % cols else image
    cv2.imshow('all', np.concatenate(row, axis=0))
    cv2.waitKey()


def drawBox(img, p1, p2, label=None, color=None, line_thickness=None):
    if img.dtype == np.uint8:
        color = color or [random.randint(0, 255) for _ in range(3)]
    else:
        color = color or [random.randint(0, 255) / 255 for _ in range(3)]
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    cv2.rectangle(img, p1, p2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        p2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (p1[0], p1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
