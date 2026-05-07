# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_comparison_figure(image_list, output_path, img_suffix):
    """
    生成 3行3列 (九宫格) 的对比大图
    """
    # 设置画布: 3行3列 (宽18, 高15 保证九宫格的清晰度和比例)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
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
            ax.set_title(title, y=1.02, fontsize=24, fontweight='bold')
        else:
            print(f"⚠️ 找不到文件: {file_path}")
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', color='red', fontsize=20)
            ax.set_title(title, y=1.02, fontsize=24, fontweight='bold')

        # 移除坐标轴
        ax.axis('off')

    # 调整布局 (紧凑一点)
    plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=0.5)
    
    # 自动生成对应的 PDF 文件名 (学术排版强烈推荐使用 PDF)
    pdf_output_path = output_path.replace('.png', '.pdf')
    
    # 保存 (同时输出 png 和 pdf)
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ 论文大图已生成: {output_path}")
    print(f"✅ 矢量大图已生成: {pdf_output_path} (请直接在 LaTeX 里引用此文件)")

# =========================================================================
#  配置区域
# =========================================================================
if __name__ == '__main__':
    target_img_name = "00706N.png" # 你的图片后缀

    # 定义 9 张图的顺序 (3行3列)
    # 格式: ("论文中显示的标题", "文件名前缀")
    papers_list = [
        # --- Row 1 (原图与最经典的对比) ---
        ("Infrared",    "Infrared"),
        ("Visible",     "Visible"),
        ("DenseFuse",   "DenseFuse"),
        
        # --- Row 2 (其他方法) ---
        ("FusionGAN",   "FusionGAN"),
        ("DIDFuse",     "DIDFuse"),
        ("U2Fusion",    "U2Fusion"),
        
        # --- Row 3 (较新方法与 Ours) ---
        ("SDNet",       "SDNet"),
        ("DeFusion",    "DeFusion"),
        ("ImprovedCDDFuse", "Ours"),   # 你的模型放在最后压轴，文件名为 Ours
    ]

    save_path = "Figure_Comparison_3x3.png" # 输出文件名

    create_comparison_figure(papers_list, save_path, target_img_name)