import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img, w):
        image_y = image_vis[:, :1, :, :]
        
        # ====== 终极防爆锁：死死卡住 w 的范围 ======
        w = torch.clamp(w, min=0.0, max=1.0)
        # ============================================
        
        # 动态强度损失
        loss_in = torch.mean(w * torch.abs(generate_img - image_y) + 
                             (1.0 - w) * torch.abs(generate_img - image_ir))
        
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        
        loss_total = loss_in + 10 * loss_grad
        return loss_total, loss_in, loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

# =======================================================
# 补回缺失的 cc 函数 (Phase I 重建特征约束必须用到)
# =======================================================
def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
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
    
    cc_val = numerator / (denominator + eps)
    cc_val = torch.clamp(cc_val, -1., 1.)
    return cc_val.mean()