# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import sys
import cv2  # 引入 opencv 处理颜色

sys.path.append(os.getcwd())

# 引入组件
from net_hf_pro import Restormer_Encoder, BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder_HF_Pro
# 移除了 Evaluator
# from utils.Evaluator import Evaluator
# from utils.img_read_save import img_save, image_read_cv2

# ================= 配置区域 =================
# 指向 train_hf_pro.py 训练出的模型
ckpt_path = "models/basehf_pro.pth"       
result_name = "results_hf_pro_train"            

test_folder = "MSRS_train" 
# ============================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")
    
    test_out_folder = os.path.join(os.getcwd(), result_name)
    os.makedirs(test_out_folder, exist_ok=True)

    # 1. 初始化模型
    print("⚙️ 初始化增强版高频注入模型 (HF Pro)...")
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder_HF_Pro()).to(device) 
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    # 2. 加载权重
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误：找不到模型文件 {ckpt_path}")
        return
    
    print(f"📥 加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 容错加载逻辑
    try:
        Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
        Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
        BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer'])
        DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
    except KeyError:
        from collections import OrderedDict
        def remove_prefix(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            return new_state_dict
        Encoder.load_state_dict(remove_prefix(checkpoint['DIDF_Encoder']))
        Decoder.load_state_dict(remove_prefix(checkpoint['DIDF_Decoder']))
        BaseFuseLayer.load_state_dict(remove_prefix(checkpoint['BaseFuseLayer']))
        DetailFuseLayer.load_state_dict(remove_prefix(checkpoint['DetailFuseLayer']))

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    # 3. 路径准备
    if os.path.exists(os.path.join(test_folder, "ir")):
        ir_folder = os.path.join(test_folder, "ir")
        vi_folder = os.path.join(test_folder, "vi")
    else:
        ir_folder = os.path.join(test_folder, "Ir")
        vi_folder = os.path.join(test_folder, "Vi")

    if not os.path.exists(ir_folder):
        print(f"❌ 错误：找不到测试集文件夹 {test_folder}")
        return

    img_list = [x for x in os.listdir(ir_folder) if x.lower().endswith(('.png', '.jpg', '.bmp', '.tif'))]
    img_list.sort()
    
    print(f"📸 开始融合 {len(img_list)} 张图片 (HF Pro - 彩色模式)...")
    
    # ================= 阶段一：生成彩色融合图像 =================
    with torch.no_grad():
        for i, img_name in enumerate(img_list):
            vi_path = os.path.join(vi_folder, img_name)
            ir_path = os.path.join(ir_folder, img_name)

            # 1. 彩色读取 Visible (BGR -> YCrCb)
            img_vi_bgr = cv2.imread(vi_path)
            if img_vi_bgr is None:
                print(f"❌ 无法读取: {vi_path}")
                continue

            img_vi_ycrcb = cv2.cvtColor(img_vi_bgr, cv2.COLOR_BGR2YCrCb)
            
            # 分离通道
            vi_Y = img_vi_ycrcb[:, :, 0]  # 亮度 (用于融合)
            vi_Cr = img_vi_ycrcb[:, :, 1] # 色彩 (用于回填)
            vi_Cb = img_vi_ycrcb[:, :, 2] # 色彩 (用于回填)

            # 2. 灰度读取 Infrared
            img_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

            # 3. 归一化 & 转 Tensor
            # data_VIS 使用 Y 通道
            data_VIS = vi_Y[np.newaxis, np.newaxis, ...] / 255.0
            data_IR = img_ir_gray[np.newaxis, np.newaxis, ...] / 255.0

            data_IR = torch.FloatTensor(data_IR).to(device)
            data_VIS = torch.FloatTensor(data_VIS).to(device)

            # 4. 推理
            feature_V_B, feature_V_D, _ = Encoder(data_VIS)
            feature_I_B, feature_I_D, _ = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            
            # Decoder_HF_Pro 前向传播
            # data_VIS (Y通道) -> inp_img (保 SSIM)
            # data_IR  -> ir_img  (提 SF, 1.5x 注入)
            data_Fuse, _ = Decoder(data_VIS, data_IR, feature_F_B, feature_F_D)
            
            # 5. 后处理与颜色回填
            data_Fuse = torch.clamp(data_Fuse, 0, 1)
            # 转回 0-255 范围
            fused_Y = np.squeeze(data_Fuse.cpu().numpy()) * 255.0
            
            # 创建 YCrCb 容器
            fused_ycrcb = np.zeros_like(img_vi_ycrcb)
            fused_ycrcb[:, :, 0] = fused_Y      # 融合后的亮度
            fused_ycrcb[:, :, 1] = vi_Cr        # 原始可见光色彩
            fused_ycrcb[:, :, 2] = vi_Cb        # 原始可见光色彩
            
            # 转回 BGR (OpenCV 默认保存格式)
            fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # 保存
            save_path = os.path.join(test_out_folder, img_name)
            cv2.imwrite(save_path, fused_bgr)
            
            sys.stdout.write(f'\r正在融合: {i+1}/{len(img_list)}')
            sys.stdout.flush()

    print(f"\n✅ 全部完成！彩色结果已保存至 {test_out_folder}")

if __name__ == '__main__':
    main()