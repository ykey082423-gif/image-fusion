# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch

# 确保引入我们新写的“跨模态热引导融合”类
from net_cbam import Restormer_Encoder, ThermalGuidedBaseFusion, ThermalGuidedDetailFusion
from net_hf_pro import Restormer_Decoder_HF_Pro

# ================= 配置区域 =================
# 1. 替换为你刚刚跑出来的最新模型
ckpt_path = "models/Thermal_Adaptive_MSRS_Fast_final.pth"

# 2. 测试集路径 (论文策略：你可以跑完MSRS后，把路径改成M3FD再跑一次，做跨数据集泛化证明)
test_vis_folder = "/home/yelei/shiyan26427/M3FD_test/M3FD_Fusion/Vis" # 记得根据你电脑的实际路径修改
test_ir_folder = "/home/yelei/shiyan26427/M3FD_test/M3FD_Fusion/Ir"  # 记得根据你电脑的实际路径修改

# 3. 结果保存路径
save_path = "results_thermal_adaptive_fast_m3fd" # 论文里可以叫 "Thermal_Adaptive_M3FD_Fast"，记得改成你模型名称的版本
# ============================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_path, exist_ok=True)

    # 1. 初始化最新的网络结构（对齐咱们工作点1的创新）
    Encoder = Restormer_Encoder().to(device)
    Decoder = Restormer_Decoder_HF_Pro().to(device)
    BaseFuseLayer = ThermalGuidedBaseFusion(dim=64, num_heads=8).to(device)
    DetailFuseLayer = ThermalGuidedDetailFusion(dim=64, num_layers=1).to(device)

    # 2. 加载权重
    print(f"📥 正在加载 SCI 冲刺版模型权重: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    def remove_prefix(d):
        return {k.replace('module.', ''): v for k, v in d.items()}

    Encoder.load_state_dict(remove_prefix(state_dict['DIDF_Encoder']), strict=False)
    Decoder.load_state_dict(remove_prefix(state_dict['DIDF_Decoder']), strict=False)
    BaseFuseLayer.load_state_dict(remove_prefix(state_dict['BaseFuseLayer']), strict=False)
    DetailFuseLayer.load_state_dict(remove_prefix(state_dict['DetailFuseLayer']), strict=False)
    
    Encoder.eval(); Decoder.eval(); BaseFuseLayer.eval(); DetailFuseLayer.eval()

    img_list = sorted(os.listdir(test_vis_folder))
    print(f"🚀 开始测试，共 {len(img_list)} 张图片...")

    with torch.no_grad():
        for img_name in img_list:
            vis_path = os.path.join(test_vis_folder, img_name)
            ir_path = os.path.join(test_ir_folder, img_name)
            if not os.path.exists(ir_path): continue

            # 读取图片并转换色彩空间
            img_vi_bgr = cv2.imread(vis_path)
            img_vi_ycrcb = cv2.cvtColor(img_vi_bgr, cv2.COLOR_BGR2YCrCb)
            vi_y = img_vi_ycrcb[:, :, 0]  
            vi_cr = img_vi_ycrcb[:, :, 1]  
            vi_cb = img_vi_ycrcb[:, :, 2]  

            img_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

            # 转成Tensor
            vi_tensor = torch.FloatTensor(vi_y).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            ir_tensor = torch.FloatTensor(img_ir_gray).unsqueeze(0).unsqueeze(0).to(device) / 255.0

            # 提取特征
            f_v_b, f_v_d, _ = Encoder(vi_tensor)
            f_i_b, f_i_d, _ = Encoder(ir_tensor)
            
            # 【重要改动】传入双模态特征，让红外特征生成掩码去引导可见光融合
            f_f_b = BaseFuseLayer(f_v_b, f_i_b)
            f_f_d = DetailFuseLayer(f_v_d, f_i_d)
            
            # 【重要改动】解码，咱们的 Decoder 现在会返回 (fused_tensor, w) 两个值
            fused_tensor, w = Decoder(vi_tensor, ir_tensor, f_f_b, f_f_d)
            
            # 还原回图像
            fused_y = (fused_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            vi_cr = cv2.resize(vi_cr, (fused_y.shape[1], fused_y.shape[0]))
            vi_cb = cv2.resize(vi_cb, (fused_y.shape[1], fused_y.shape[0]))

            result_ycrcb = cv2.merge([fused_y, vi_cr, vi_cb])
            result_bgr = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)

            cv2.imwrite(os.path.join(save_path, img_name), result_bgr)
            print(f"已生成: {img_name}")

    print(f"✅ 测试完毕！结果保存在文件夹: {save_path}")

if __name__ == '__main__':
    main()