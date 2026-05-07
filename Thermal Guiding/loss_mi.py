import torch
import torch.nn as nn

class MILoss(nn.Module):
    def __init__(self):
        super(MILoss, self).__init__()

    def forward(self, image_F, image_A, image_B):
        """
        计算互信息损失 (Mutual Information Loss)
        原理：最大化 MI 等价于最大化融合图与源图像的相关性。
        为了作为 Loss (最小化)，我们计算: (1 - CC_FA) + (1 - CC_FB)
        """
        loss_mi = (1 - self.cc(image_F, image_A)) + (1 - self.cc(image_F, image_B))
        return loss_mi

    def cc(self, img1, img2):
        """
        计算两个图像张量之间的相关系数 (Correlation Coefficient)
        """
        eps = torch.finfo(torch.float32).eps
        N, C, _, _ = img1.shape
        
        # Flatten
        img1 = img1.reshape(N, C, -1)
        img2 = img2.reshape(N, C, -1)
        
        # 减去均值 (Center the data)
        img1 = img1 - img1.mean(dim=-1, keepdim=True)
        img2 = img2 - img2.mean(dim=-1, keepdim=True)
        
        # 计算 CC
        numerator = torch.sum(img1 * img2, dim=-1)
        denominator = torch.sqrt(torch.sum(img1 ** 2, dim=-1)) * torch.sqrt(torch.sum(img2 ** 2, dim=-1))
        
        cc = numerator / (denominator + eps)
        cc = torch.clamp(cc, -1., 1.)
        return cc.mean()