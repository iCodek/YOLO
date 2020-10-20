import cv2
import os

root = r'C:\Users\85127\Downloads\qrimg\test'
out = r'C:\Users\85127\Downloads\qrimg\bmp'
imglist = os.listdir(root)
print(imglist)
for img in imglist:
    rgb = cv2.imread(os.path.join(root, img), -1)
    gary = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
    _, gary = cv2.threshold(gary,120,255,cv2.THRESH_OTSU)
    #cv2.imshow('x',gary)
    cv2.imwrite(os.path.join(out, img.replace('.bmp','.bmp').replace('.jpg','.bmp')), gary)