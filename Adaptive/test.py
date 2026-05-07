# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from net_cbam import Restormer_Encoder, BaseFeatureExtraction, DetailFeatureExtraction
from net_hf_pro import Restormer_Decoder_HF_Pro

# ================= 配置区域 =================
ckpt_path = "models/Adaptive_GAHI_Net_epoch_119.pth"
test_vis_folder = "/home/yelei/shiyan1/M3FD_test/M3FD_Fusion/Vis"
test_ir_folder = "/home/yelei/shiyan1/M3FD_test/M3FD_Fusion/Ir"
save_path = "results_adaptive_m3fd3"
# ============================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_path, exist_ok=True)

    # 1. 初始化模型
    Encoder = Restormer_Encoder().to(device)
    Decoder = Restormer_Decoder_HF_Pro().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction().to(device)

    # 2. 【核心修复】正确拆解并加载 4 个网络的权重
    print(f"📥 正在完整加载 4 个模块的模型权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    def remove_prefix(d):
        return {k.replace('module.', ''): v for k, v in d.items()}

    Encoder.load_state_dict(remove_prefix(checkpoint['DIDF_Encoder']), strict=False)
    Decoder.load_state_dict(remove_prefix(checkpoint['DIDF_Decoder']), strict=False)
    BaseFuseLayer.load_state_dict(remove_prefix(checkpoint['BaseFuseLayer']), strict=False)
    DetailFuseLayer.load_state_dict(remove_prefix(checkpoint['DetailFuseLayer']), strict=False)
    
    Encoder.eval(); Decoder.eval(); BaseFuseLayer.eval(); DetailFuseLayer.eval()

    img_list = sorted(os.listdir(test_vis_folder))
    print(f"🚀 开始 YCbCr 模式测试，共 {len(img_list)} 张图片...")

    with torch.no_grad():
        for img_name in img_list:
            vis_path = os.path.join(test_vis_folder, img_name)
            ir_path = os.path.join(test_ir_folder, img_name)
            if not os.path.exists(ir_path): continue

            # --- A. YCbCr 颜色处理逻辑 ---
            img_vi_bgr = cv2.imread(vis_path)
            # 转为 YCrCb
            img_vi_ycrcb = cv2.cvtColor(img_vi_bgr, cv2.COLOR_BGR2YCrCb)
            vi_y = img_vi_ycrcb[:, :, 0]   # 亮度通道 (用于融合)
            vi_cr = img_vi_ycrcb[:, :, 1]  # 颜色通道 (保留)
            vi_cb = img_vi_ycrcb[:, :, 2]  # 颜色通道 (保留)

            # --- B. 读取红外 (灰度) ---
            img_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

            # --- C. 转为 Tensor ---
            vi_tensor = torch.FloatTensor(vi_y).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            ir_tensor = torch.FloatTensor(img_ir_gray).unsqueeze(0).unsqueeze(0).to(device) / 255.0

            # --- D. 推理 ---
            f_v_b, f_v_d, _ = Encoder(vi_tensor)
            f_i_b, f_i_d, _ = Encoder(ir_tensor)
            f_f_b = BaseFuseLayer(f_v_b + f_i_b)
            f_f_d = DetailFuseLayer(f_v_d + f_i_d)
            
            fused_tensor, w = Decoder(vi_tensor, ir_tensor, f_f_b, f_f_d)
            
            # --- E. 结果还原与颜色回填 ---
            fused_y = (fused_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # 必须保证尺寸一致 (防止对齐问题)
            vi_cr = cv2.resize(vi_cr, (fused_y.shape[1], fused_y.shape[0]))
            vi_cb = cv2.resize(vi_cb, (fused_y.shape[1], fused_y.shape[0]))

            # 合并回 YCrCb 并转回 BGR 彩色图
            result_ycrcb = cv2.merge([fused_y, vi_cr, vi_cb])
            result_bgr = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

            # --- F. 保存 ---
            cv2.imwrite(os.path.join(save_path, img_name), result_bgr)
            
            # 打印当前动态权重
            print(f"已生成: {img_name} | 当前环境权重 w: {w.mean().item():.4f}")

    print(f"✅ 搞定！彩色融合图已保存至: {save_path}")

if __name__ == '__main__':
    main()