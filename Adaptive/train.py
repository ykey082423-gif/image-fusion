# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from net_cbam import Restormer_Encoder, BaseFeatureExtraction, DetailFeatureExtraction
from net_hf_pro import Restormer_Decoder_HF_Pro
from utils.dataset import M3FD_Dataset
from utils.loss import Fusionloss, cc
import kornia

# ================= 配置区域 =================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

criteria_fusion = Fusionloss()
model_str = 'Adaptive_GAHI_Net'

# ⚠️ 严格对齐基线
num_epochs = 120 
epoch_gap = 40  
lr = 5e-5           
weight_decay = 0
batch_size = 2      

# ⚠️ 恢复了所有基线必备的 Loss 权重
coeff_mse_loss_VF = 20. 
coeff_mse_loss_IF = 20.
coeff_decomp = 2.
coeff_grad_max = 20. 

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = M3FD_Dataset(vis_dir="M3FD_train/vi", ir_dir="M3FD_train/ir", patch_size=256)
# 加上 drop_last=True 防止结尾报错
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True) 

# ================= 初始化模型 =================
# 单卡不使用 DataParallel 规避 Bug
Encoder = Restormer_Encoder().cuda()
Decoder = Restormer_Decoder_HF_Pro().cuda() 
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).cuda()
DetailFuseLayer = DetailFeatureExtraction(num_layers=1).cuda()

optimizer1 = torch.optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)
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
print("🚀 开始自适应版严格控制变量训练 (M3FD数据集)...")

# 循环开始
for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(trainloader):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        
        # 防护服：强行清理脏数据
        data_VIS = torch.nan_to_num(data_VIS, nan=0.0, posinf=1.0, neginf=0.0)
        data_IR = torch.nan_to_num(data_IR, nan=0.0, posinf=1.0, neginf=0.0)
        
        Encoder.train(); Decoder.train(); BaseFuseLayer.train(); DetailFuseLayer.train()
        optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()

        # ================= Phase I：独立重建期 =================
        if epoch < epoch_gap: 
            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            
            # 使用新版兼容 None 的 Decoder
            data_VIS_hat, _ = Decoder(data_VIS, None, feature_V_B, feature_V_D)
            data_IR_hat, _ = Decoder(None, data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
            Gradient_loss = L1Loss(spatial_grad(data_VIS), spatial_grad(data_VIS_hat))
            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + coeff_decomp * loss_decomp + 5.0 * Gradient_loss
            loss.backward()
            nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step(); optimizer2.step()

        # ================= Phase II：动态融合期 =================
        else:  
            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            
            # 取出动态融合图 data_Fuse，和权重 w
            data_Fuse, w = Decoder(data_VIS, data_IR, feature_F_B, feature_F_D)  

            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            
            # 使用原生 Fusionloss
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse, w)
            
            # 恢复丢失的梯度最大化约束 (非常重要！)
            grad_vis = torch.abs(spatial_grad(data_VIS))
            grad_ir = torch.abs(spatial_grad(data_IR))
            grad_max = torch.max(grad_vis, grad_ir)
            grad_fuse = torch.abs(spatial_grad(data_Fuse))
            loss_grad_max = L1Loss(grad_fuse, grad_max)
            
            loss = fusionloss + coeff_decomp * loss_decomp + coeff_grad_max * loss_grad_max
            
            loss.backward()
            nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step(); optimizer2.step(); optimizer3.step(); optimizer4.step()

        # ================= 终端实时打印进度 =================
        batches_done = epoch * len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 10 == 0:
            if epoch < epoch_gap:
                # 前40轮不显示 w
                sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}] [Iter {i}/{len(trainloader)}] [loss: {loss.item():.4f}] ETA: {time_left}")
            else:
                # 40轮后显示动态权重 w
                sys.stdout.write(f"\r[Epoch {epoch}/{num_epochs}] [Iter {i}/{len(trainloader)}] [loss: {loss.item():.4f}] [w: {w.mean().item():.3f}] ETA: {time_left}")
            sys.stdout.flush()

    scheduler1.step(); scheduler2.step()
    if epoch >= epoch_gap:
        scheduler3.step(); scheduler4.step()

# ================= 退出循环后：只保存一次！ =================
checkpoint = {
    'DIDF_Encoder': Encoder.state_dict(),
    'DIDF_Decoder': Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}
os.makedirs("models", exist_ok=True)
save_name = os.path.join("models", f"{model_str}_epoch_119.pth")
torch.save(checkpoint, save_name)
print(f"\n✅ 训练结束！模型已保存至 {save_name}")