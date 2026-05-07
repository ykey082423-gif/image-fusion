# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_comparison_figure(image_list, output_path, img_suffix):
    """
    生成 2行5列 的对比大图
    """
    # 设置画布: 2行5列 (宽25, 高8 保证清晰度和比例)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 9))
    axes = axes.flatten()

    # 字体设置 (Times New Roman)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    base_dir = "paper_figures" # 图片所在文件夹

    for i, (title, prefix) in enumerate(image_list):
        # 构造文件名
        filename = f"{prefix}_{img_suffix}"
        file_path = os.path.join(base_dir, filename)

        ax = axes[i]
        
        if os.path.exists(file_path):
            img = Image.open(file_path)
            ax.imshow(img)
            # 设置标题 (加粗，字号大一点)
            ax.set_title(title, y=1.01, fontsize=24, fontweight='bold')
        else:
            print(f"⚠️ 找不到文件: {file_path}")
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', color='red')

        # 移除坐标轴
        ax.axis('off')

    # 调整布局 (紧凑一点)
    plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.5)
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 论文大图已生成: {output_path}")

# =========================================================================
#  配置区域
# =========================================================================
if __name__ == '__main__':
    target_img_name = "00706N.png" # 后缀名

    # 定义 10 张图的顺序 (2行5列)
    # 格式: ("论文中显示的标题", "文件名前缀")
    papers_list = [
        # --- Row 1 (参考 CDDFuse 排列) ---
        ("Infrared",    "Infrared"),
        ("Visible",     "Visible"),
        ("DIDFuse",     "DIDFuse"),
        ("U2Fusion",    "U2Fusion"),
        ("SDNet",       "SDNet"),
        
        # --- Row 2 ---
        ("DenseFuse",   "DenseFuse"),
        ("DeFusion",    "DeFusion"),
        ("FusionGAN",     "FusionGAN"),
        ("CDDFuse",     "CDDFuse"),   # 强力竞品放在你前面
        ("GAHINet (Ours)", "Ours"),   # 你的模型放在最后压轴，文件名为 Ours
    ]

    save_path = "Figure_Comparison_10.png" # 输出文件名

    create_comparison_figure(papers_list, save_path, target_img_name)