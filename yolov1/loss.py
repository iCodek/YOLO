import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable


class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5, classNum=20):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.classNum = classNum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_iou(self, bbox1, bbox2):
        """
        Compute the intersection over union of two set of boxes, each box is [x_center,y_center,w,h]
        :param bbox1: (tensor) bounding boxes, size [N,4]
        :param bbox2: (tensor) bounding boxes, size [M,4]
        :return:
        """
        # transfer center cordinate to x1,y1,x2,y2
        b1x1y1 = bbox1[:, :2] - bbox1[:, 2:] ** 2  # [N, (x1,y1)=2]
        b1x2y2 = bbox1[:, :2] + bbox1[:, 2:] ** 2  # [N, (x2,y2)=2]
        b2x1y1 = bbox2[:, :2] - bbox2[:, 2:] ** 2  # [M, (x1,y1)=2]
        b2x2y2 = bbox2[:, :2] + bbox2[:, 2:] ** 2  # [M, (x1,y1)=2]
        box1 = torch.cat((b1x1y1.view(-1, 2), b1x2y2.view(-1, 2)), dim=1)  # [N,4], 4=[x1,y1,x2,y2]
        box2 = torch.cat((b2x1y1.view(-1, 2), b2x2y2.view(-1, 2)), dim=1)  # [M,4], 4=[x1,y1,x2,y2]
        N = box1.size(0)
        M = box2.size(0)
        # find cordinate of intersaction boxes.
        tl = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        br = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        #   width and height
        wh = br - tl  # [N,M,2]
        wh[(wh < 0).detach()] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred, target):
        """
        :param pred: [batch,SxSx(Bx5+20))]
        :param target: [batch,SxSx(Bx5+20))]
        :return: Yolov1Loss
        """
        S = target.size(1)
        B = (target.size(-1) - self.classNum) // 5
        obj = target[:, :, :, 4] > 0
        noobj = target[:, :, :, 4] == 0


        loss1_B1 = torch.square(pred[:,:,:,0]-target[:,:,:,0])+torch.square(pred[:,:,:,1]-target[:,:,:,1])
        loss1_B1 = loss1_B1 * obj

        loss1_B2 = torch.square(pred[:, :, :, 5] - target[:, :, :, 5]) + torch.square(
            pred[:, :, :, 6] - target[:, :, :, 6])
        loss1_B2 = loss1_B2 * obj

        B_chosen = torch.zeros((target.size(0), target.size(1), target.size(2), 2)).to(self.device)
        for b in range(target.size(0)):
            for r in range(target.size(1)):
                for c in range(target.size(2)):
                    box1 = torch.cat([pred[b, r, c, :4].unsqueeze(0), pred[b, r, c, 5:9].unsqueeze(0)], dim=0)
                    box2 = target[b, r, c, :4].unsqueeze(0)
                    iou = self.compute_iou(box1, box2)
                    max_iou, max_index = iou.max(0)
                    B_chosen[b,r,c,max_index] = 1

        loss1 = loss1_B1*B_chosen[:,:,:,0] + loss1_B2*B_chosen[:,:,:,1]
        loss1 = torch.sum(loss1)

        loss2_B1 = torch.square(torch.sqrt(pred[:,:,:,2])-torch.sqrt(target[:,:,:,2]))+torch.square(torch.sqrt(pred[:,:,:,3])-torch.sqrt(target[:,:,:,3]))
        loss2_B1 = loss2_B1 * obj

        loss2_B2 = torch.square(torch.sqrt(pred[:,:,:,7])-torch.sqrt(target[:,:,:,7]))+torch.square(torch.sqrt(pred[:,:,:,8])-torch.sqrt(target[:,:,:,8]))
        loss2_B2 = loss2_B2 * obj

        loss2 = loss2_B1 * B_chosen[:, :, :, 0] + loss2_B2 * B_chosen[:, :, :, 1]
        loss2 = torch.sum(loss2)


        loss3_B1 = torch.square(pred[:,:,:,4]-target[:,:,:,4]) * obj
        loss3_B2 = torch.square(pred[:, :, :, 9] - target[:, :, :, 9]) * obj
        loss3 = loss3_B1 * B_chosen[:, :, :, 0] + loss3_B2 * B_chosen[:, :, :, 1]
        loss3 = torch.sum(loss3)

        loss4_B1 = torch.square(pred[:, :, :, 4] - target[:, :, :, 4]) * noobj
        loss4_B2 = torch.square(pred[:, :, :, 9] - target[:, :, :, 9]) * noobj
        loss4 = loss4_B1 * B_chosen[:, :, :, 0] + loss4_B2 * B_chosen[:, :, :, 1]
        loss4 = torch.sum(loss4)

        loss5 = torch.sum(torch.square(pred[:,:,:,10:] - target[:,:,:,10:]),dim=3) * obj
        loss5 = torch.sum(loss5)

        return  self.lambda_coord* loss1 +\
                self.lambda_coord*loss2 +\
                loss3 + \
                self.lambda_noobj*loss4 + loss5

        # B_chosen = torch.zeros((target.size(0), target.size(1), target.size(2), 26)).to(self.device)
        # for b in range(target.size(0)):
        #     for r in range(target.size(1)):
        #         for c in range(target.size(2)):
        #             box1 = torch.cat([pred[b, r, c, :4].unsqueeze(0), pred[b, r, c, 5:9].unsqueeze(0)], dim=0)
        #             box2 = target[b, r, c, :4].unsqueeze(0)
        #             iou = self.compute_iou(box1, box2)
        #             max_iou, max_index = iou.max(0)
        #             if target[b, r, c, 4] > 0:
        #                 B_chosen[b, r, c, :4] = pred[b, r, c, 5 * max_index:5 * max_index + 4]
        #                 B_chosen[b, r, c, 4] = pred[b, r, c, 4]
        #                 B_chosen[b, r, c, 6:] = pred[b, r, c, 10:]
        #                 B_chosen[b, r, c, 5] = 1
        #             else:
        #                 B_chosen[b, r, c, 5] = pred[b, r, c, 4]
        # loss1 = functional.mse_loss(B_chosen[:, :, :, 0], target[:, :, :, 0], reduction='sum') + \
        #         functional.mse_loss(B_chosen[:, :, :, 1], target[:, :, :, 1], reduction='sum')
        # loss2 = functional.mse_loss(torch.sqrt(B_chosen[:, :, :, 2]), torch.sqrt(target[:, :, :, 2]), reduction='sum') + \
        #         functional.mse_loss(torch.sqrt(B_chosen[:, :, :, 3]), torch.sqrt(target[:, :, :, 3]), reduction='sum')
        # loss3 = functional.mse_loss(B_chosen[:, :, :, 4], target[:, :, :, 4], reduction='sum')
        # loss4 = functional.mse_loss(B_chosen[:, :, :, 5], target[:, :, :, 4], reduction='sum')
        # loss5 = functional.mse_loss(B_chosen[:, :, :, 6:], target[:, :, :, 10:], reduction='sum')
        # print(B_chosen[:, :, :, 0], target[:, :, :, 0])
        # print(loss1, loss2, loss3, loss4, loss5)
        #
        # loss = self.lambda_coord * loss1 + self.lambda_coord * loss2 + loss3 + self.lambda_noobj * loss4 + loss5
        #return torch.true_divide(loss, target.size(0))

        # box1 = torch.tensor([[1,1,2,1],
        #                      [1,1,4,4]])
        # box2 = torch.tensor([[1,1,2,2]])
        # print(self.compute_iou(box1, box2))

    def compute_iou(self, box1, box2):
        """
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = torch.true_divide(inter, area1 + area2 - inter)
        return iou
