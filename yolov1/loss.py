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

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def forward(self, pred, target):
        """
        :param pred: [batch,SxSx(Bx5+20))]
        :param target: [batch,SxSx(Bx5+20))]
        :return: Yolov1Loss
        """
        # S = target.size(1)
        # B = (target.size(-1) - self.classNum) // 5
        # obj = target[:, :, :, 4] > 0
        # noobj = target[:, :, :, 4] == 0
        B_chosen = torch.zeros((target.size(0), target.size(1), target.size(2), 26)).to(self.device)
        for b in range(target.size(0)):
            for r in range(target.size(1)):
                for c in range(target.size(2)):
                    box1 = torch.cat([pred[b, r, c, :4].unsqueeze(0), pred[b, r, c, 5:9].unsqueeze(0)], dim=0)
                    box2 = target[b, r, c, :4].unsqueeze(0)
                    iou = self.compute_iou(box1, box2)
                    max_iou, max_index = iou.max(0)
                    if target[b, r, c, 4] > 0:
                        B_chosen[b, r, c, :4] = pred[b, r, c, 5 * max_index:5 * max_index + 4]
                        B_chosen[b, r, c, 4] = pred[b, r, c, 4]
                        B_chosen[b, r, c, 6:] = pred[b, r, c, 10:]
                        B_chosen[b, r, c, 5] = 1
                    else:
                        B_chosen[b, r, c, 5] = pred[b, r, c, 4]
        loss1 = functional.mse_loss(B_chosen[:, :, :, 0], target[:, :, :, 0], reduction='sum') + \
                functional.mse_loss(B_chosen[:, :, :, 1], target[:, :, :, 1], reduction='sum')
        loss2 = functional.mse_loss(torch.sqrt(B_chosen[:, :, :, 2]), torch.sqrt(target[:, :, :, 2]), reduction='sum') + \
                functional.mse_loss(torch.sqrt(B_chosen[:, :, :, 3]), torch.sqrt(target[:, :, :, 3]), reduction='sum')
        loss3 = functional.mse_loss(B_chosen[:, :, :, 4], target[:, :, :, 4], reduction='sum')
        loss4 = functional.mse_loss(B_chosen[:, :, :, 5], target[:, :, :, 4], reduction='sum')
        loss5 = functional.mse_loss(B_chosen[:, :, :, 6:], target[:, :, :, 10:], reduction='sum')

        loss = self.lambda_coord * loss1 + self.lambda_coord * loss2 + loss3 + self.lambda_noobj * loss4 + loss5
        return torch.true_divide(loss, target.size(0))

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
