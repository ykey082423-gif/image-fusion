import os
import numpy as np
from utils.Evaluator import Evaluator, image_read_cv2

# ================= 路径配置 =================
fused_dir = "results_thermal_adaptive_fast_m3fd" # 刚刚融合出图的文件夹
vis_dir = "/home/yelei/shiyan26427/M3FD_test/M3FD_Fusion/Vis"                    # MSRS可见光测试集
ir_dir = "/home/yelei/shiyan26427/M3FD_test/M3FD_Fusion/Ir"                     # MSRS红外测试集
# ============================================

def main():
    img_list = sorted(os.listdir(fused_dir))
    num_imgs = len(img_list)
    if num_imgs == 0:
        print("融合文件夹为空，请先运行 test.py 生成图片！")
        return

    print(f"📊 开始计算客观指标，共 {num_imgs} 张图片...")
    
    # 初始化累加器
    total_MI, total_SSIM, total_Qabf, total_SF, total_SCD = 0, 0, 0, 0, 0

    for i, img_name in enumerate(img_list):
        fused_path = os.path.join(fused_dir, img_name)
        vis_path = os.path.join(vis_dir, img_name)
        ir_path = os.path.join(ir_dir, img_name)

        # 统一读取为灰度图进行指标计算 (Evaluator 的规范)
# 统一读取为灰度图进行指标计算，保持 0-255 的离散整数
        imgF = image_read_cv2(fused_path, mode='GRAY')
        imgA = image_read_cv2(vis_path, mode='GRAY')
        imgB = image_read_cv2(ir_path, mode='GRAY')

        # 计算 SCI 论文最爱看的 5 大核心指标
        total_MI += Evaluator.MI(imgF, imgA, imgB)
        total_SSIM += Evaluator.SSIM(imgF, imgA, imgB)
        total_Qabf += Evaluator.Qabf(imgF, imgA, imgB)
        total_SF += Evaluator.SF(imgF)
        total_SCD += Evaluator.SCD(imgF, imgA, imgB)

        if (i+1) % 50 == 0:
            print(f"已计算 {i+1}/{num_imgs} 张...")

    # 打印最终平均结果
    print("\n" + "="*40)
    print("🏆 测试集客观指标最终平均分 (写进论文表格里)：")
    print(f"1. MI (互信息 - 信息保留度)   : {total_MI / num_imgs:.4f}")
    print(f"2. SSIM (结构相似度)          : {total_SSIM / num_imgs:.4f}")
    print(f"3. Qabf (边缘信息保留)        : {total_Qabf / num_imgs:.4f}")
    print(f"4. SF (空间频率 - 清晰度)     : {total_SF / num_imgs:.4f}")
    print(f"5. SCD (差异相关性总和)       : {total_SCD / num_imgs:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()