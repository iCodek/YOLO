from util.download import checkFile
from util.convert import xmlConvert
from util.visualize import view

# 下载解压数据集
#checkFile('VOCtrainval.tar')
# xml数据集转为Yolo格式
#xmlConvert('VOCtrainval/VOCdevkit/VOC2012/ImageSets/Main/train.txt', 'VOCtrainval/VOCdevkit/VOC2012/Annotations', 'VOC2012/label')
# 可视化
view('VOCtrainval/VOCdevkit/VOC2012/JPEGImages', 'VOC2012/label', 20)
# VOCtrainval/VOCdevkit/VOC2012/JPEGImages\2008_007161.jpg
# VOCtrainval/VOCdevkit/VOC2012/JPEGImages\2010_001924.jpg