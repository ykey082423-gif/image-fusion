# -*- coding: utf-8 -*-
import os
import numpy as np
from utils.Evaluator import Evaluator, image_read_cv2

def evaluate_group(folder_path, group_name):
    if not os.path.exists(folder_path):
        print(f"找不到文件夹: {folder_path}")
        return

    # 获取所有基础图片名称 (例如 '00725.png'，根据你的规则，这是可见光原图)
    img_list = os.listdir(folder_path)
    base_names = [f for f in img_list if f.endswith('.png') and '(' not in f]
    
    if len(base_names) == 0:
        print(f"文件夹 {folder_path} 里没有找到基础命名的图片！")
        return
        
    print(f"\n================ 正在评估: {group_name}组 ({len(base_names)}组图片) ================")
    
    sf_list, mi_list, vif_list, qabf_list = [], [], [], []
    
    for base_img in base_names:
        base_name = base_img.split('.')[0] # 获取前缀，比如 '00725'
        
        # ⚠️ 已经严格按照你的命名规则进行了匹配
        vi_path = os.path.join(folder_path, f"{base_name}.png")          # 无括号的是可见光
        ir_path = os.path.join(folder_path, f"{base_name} (2).png")      # (2) 是红外光
        fused_path = os.path.join(folder_path, f"{base_name} (3).png")   # (3) 是融合图
        
        # 检查三个文件是否都齐备
        if not (os.path.exists(vi_path) and os.path.exists(ir_path) and os.path.exists(fused_path)):
            print(f"警告: 找不到 {base_name} 对应的全套配套文件，已跳过。")
            continue
            
        # 读取图像 (灰度模式下计算指标最标准)
        img_F = image_read_cv2(fused_path, 'GRAY')
        img_VI = image_read_cv2(vi_path, 'GRAY')
        img_IR = image_read_cv2(ir_path, 'GRAY')
        
        # 计算四大核心指标
        sf_list.append(Evaluator.SF(img_F))
        mi_list.append(Evaluator.MI(img_F, img_VI, img_IR))
        vif_list.append(Evaluator.VIFF(img_F, img_VI, img_IR))
        qabf_list.append(Evaluator.Qabf(img_F, img_VI, img_IR))
        
    if sf_list:
        print(f"SF   (空间频率, 越高越清晰)   : {np.mean(sf_list):.4f}")
        print(f"MI   (互信息, 越高信息越多)   : {np.mean(mi_list):.4f}")
        print(f"VIF  (视觉保真度, 越高越自然) : {np.mean(vif_list):.4f}")
        print(f"Qabf (边缘保持度, 越高越锐利) : {np.mean(qabf_list):.4f}")

if __name__ == '__main__':
    # 注意检查路径
    evaluate_group("test/test_day", "白天 (Daytime)")
    evaluate_group("test/test_night", "黑夜/大雾 (Night/Fog)")