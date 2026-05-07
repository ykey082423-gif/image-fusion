# -*- coding: utf-8 -*-
# 完美保留你之前的网络结构！
from net_cbam import Restormer_Encoder, ThermalGuidedBaseFusion, ThermalGuidedDetailFusion
from net_hf_pro import Restormer_Decoder_HF_Pro
# 【核心提速】改回使用 MSRS 的 H5 数据集读取
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()

# 模型名称
model_str = 'Thermal_Adaptive_MSRS_Fast'

# 【极致提速关键配置】
num_epochs = 40     # 切片数据量大，网络很容易收敛，40轮完全足够支撑你的创新点和指标了
epoch_gap = 15      # 前15轮跑基础重建(Phase I)，后25轮跑热引导+自适应融合(Phase II)
lr = 5e-5           
weight_decay = 0
batch_size = 8     # 5090显存极大，直接把Batch Size拉到32！(如果万一报OOM，就退回16)

coeff_mse_loss_VF = 20. 
coeff_mse_loss_IF = 20.
coeff_decomp = 2.
coeff_grad_max = 20. 
clip_grad_norm_value = 0.01

optim_step = 20
optim_gamma = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 数据集加载 (切换为 MSRS 极速版) =================
# 请确保你的 MSRS_train.h5 文件在这个路径下
train_dataset = H5Dataset(h5file_path="MSRS_train_imgsize_128_stride_200.h5")
# 开启多个 worker，极速喂数据
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True) 

# ================= 初始化模型 (保留你的所有创新点) =================
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder_HF_Pro()).to(device) 
BaseFuseLayer = nn.DataParallel(ThermalGuidedBaseFusion(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(ThermalGuidedDetailFusion(dim=64, num_layers=1)).to(device)

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
spatial_grad = kornia.filters.SpatialGradient() 

prev_time = time.time()

print(f"🚀 启动极速版训练！包含热引导+IAM自适应。总 Iter 数仅为: {len(trainloader)} / Epoch")

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(trainloader):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        
        # 清理脏数据
        data_VIS = torch.nan_to_num(data_VIS, nan=0.0, posinf=1.0, neginf=0.0)
        data_IR = torch.nan_to_num(data_IR, nan=0.0, posinf=1.0, neginf=0.0)
        
        DIDF_Encoder.train(); DIDF_Decoder.train(); BaseFuseLayer.train(); DetailFuseLayer.train()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        # ================= Phase I =================
        if epoch < epoch_gap: 
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            
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
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value)
            optimizer1.step(); optimizer2.step()

        # ================= Phase II (热引导 + IAM) =================
        else:  
            feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
            
            feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D, feature_I_D)
            
            data_Fuse, w = DIDF_Decoder(data_VIS, data_IR, feature_F_B, feature_F_D)  

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse, w)
            
            grad_vis = torch.abs(spatial_grad(data_VIS))
            grad_ir = torch.abs(spatial_grad(data_IR))
            grad_max = torch.max(grad_vis, grad_ir)
            grad_fuse = torch.abs(spatial_grad(data_Fuse))
            loss_grad_max = L1Loss(grad_fuse, grad_max)
            
            loss = fusionloss + coeff_decomp * loss_decomp + coeff_grad_max * loss_grad_max
            
            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value)
            optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        # 终端打印信息
        if i % 10 == 0:
            batches_done = epoch * len(trainloader) + i
            batches_left = num_epochs * len(trainloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            if epoch < epoch_gap:
                sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}] [Iter {i}/{len(trainloader)}] [loss: {loss.item():.4f}] ETA: {time_left}")
            else:
                sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}] [Iter {i}/{len(trainloader)}] [loss: {loss.item():.4f}] [w: {w.mean().item():.3f}] ETA: {time_left}")
            sys.stdout.flush()

    scheduler1.step(); scheduler2.step()
    if epoch >= epoch_gap:
        scheduler3.step(); scheduler4.step()

checkpoint = {
    'DIDF_Encoder': DIDF_Encoder.state_dict(),
    'DIDF_Decoder': DIDF_Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}
os.makedirs("models", exist_ok=True)
save_name = os.path.join("models", f"{model_str}_final.pth")
torch.save(checkpoint, save_name)
print(f"\n✅ 训练成功结束！小论文核心模型已保存至 {save_name}")