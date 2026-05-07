# -*- coding: utf-8 -*-
from net_cbam import Restormer_Encoder, BaseFeatureExtraction, DetailFeatureExtraction
# 引入新的增强版 Decoder
from net_hf_pro import Restormer_Decoder_HF_Pro
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia

# ================= 配置区域 =================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'Final_HF_Pro'

num_epochs = 120 
epoch_gap = 40  
lr = 1e-4
weight_decay = 0
batch_size = 8

# === 【策略：物理注入增强 + 梯度约束】 ===
# 1. 基础约束
coeff_mse_loss_VF = 20. 
coeff_mse_loss_IF = 20.
coeff_decomp = 2.

# 2. 纹理保护组合
coeff_tv = 0.        # 禁用 TV，因为它会把我们注入的高频当噪声抹掉
coeff_grad_max = 20. # 启用 MaxGrad，强迫网络保留最强边缘 (保 Qabf)

# ===========================================

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化模型 (使用 Pro 版 Decoder)
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder_HF_Pro()).to(device) 
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
spatial_grad = kornia.filters.SpatialGradient() # 梯度计算器

trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True) 
loader = {'train': trainloader, }
prev_time = time.time()

print("🚀 开始增强版高频注入训练 (1.5x Injection + MaxGrad Loss)...")

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train(); DIDF_Decoder.train(); BaseFuseLayer.train(); DetailFuseLayer.train()
        DIDF_Encoder.zero_grad(); DIDF_Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            
            # Phase I 重建：暂时不注入，让网络先学好基础
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, None, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(None, data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
            Gradient_loss = L1Loss(spatial_grad(data_VIS), spatial_grad(data_VIS_hat))
            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + coeff_decomp * loss_decomp + 5.0 * Gradient_loss
            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step(); optimizer2.step()
        else:  #Phase II
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            
            # Phase II 融合：注入 1.5 倍高频
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, data_IR, feature_F_B, feature_F_D)  

            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            # === Max-Gradient Loss (保 Qabf/SF) ===
            grad_vis = torch.abs(spatial_grad(data_VIS))
            grad_ir = torch.abs(spatial_grad(data_IR))
            grad_max = torch.max(grad_vis, grad_ir)
            grad_fuse = torch.abs(spatial_grad(data_Fuse))
            loss_grad_max = L1Loss(grad_fuse, grad_max)
            
            loss = fusionloss + coeff_decomp * loss_decomp + coeff_grad_max * loss_grad_max
            
            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}] [loss: {loss.item():.4f}] ETA: {time_left}")

    scheduler1.step(); scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step(); scheduler4.step()

if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    os.makedirs("models", exist_ok=True)
    # 保存为 HF Pro 模型
    torch.save(checkpoint, os.path.join("models", "basehf_pro.pth"))
    print("\n✅ 训练结束！模型已保存至 models/basehf_pro.pth")