import torch
from dataset import YoLoDataset
from torch.utils.data import DataLoader
from yolov1.net import Yolov1
from yolov1.loss import YoloLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yolodataset = YoLoDataset(root='../VOCtrainval/VOCdevkit/VOC2012/JPEGImages', txt='../VOC2012/label')
trainLoader = DataLoader(yolodataset, batch_size=1, shuffle=True)
model = Yolov1()
model.train()
lossFn = YoloLoss(1)
optimizer=torch.optim.AdamW(model.parameters(), lr=0.0004,betas=(0.9,0.99))

model.to(device)
lossFn.to(device)

epochs = 10

for epoch in range(epochs):
    for i, (img, target) in enumerate(trainLoader):
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = lossFn(output, target)
        print(loss.item())
        # loss.backward()
        # optimizer.step()
        # if i % 50 == 0:
        #     print('epoch:{}, loss:{}'.format(epoch, loss.item()))