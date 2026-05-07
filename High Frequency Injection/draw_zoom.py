# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np

def draw_zoom_box(img_path, save_path, boxes):
    # 读取图片 (支持中文路径)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    
    if img is None:
        print(f"❌ 无法读取: {img_path}")
        return

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    SCALE = 2
    H, W = img.shape[:2]

    for box in boxes:
        x, y, w, h = box['loc']
        px, py = box['paste']
        color = box['color']
        thick = box['thickness']

        # --- 截取并放大 ---
        y_end = min(y+h, H)
        x_end = min(x+w, W)
        roi = img[y:y_end, x:x_end]
        
        if roi.size == 0: continue

        zoomed_w, zoomed_h = w * SCALE, h * SCALE
        zoomed = cv2.resize(roi, (zoomed_w, zoomed_h), interpolation=cv2.INTER_CUBIC)
        
        # --- 原图画框 ---
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thick)

        # --- 贴图 (自动防越界) ---
        if py + zoomed_h > H: py = H - zoomed_h - 2
        if px + zoomed_w > W: px = W - zoomed_w - 2

        img[py:py+zoomed_h, px:px+zoomed_w] = zoomed

        # --- 放大图边框 ---
        cv2.rectangle(img, (px, py), (px+zoomed_w, py+zoomed_h), color, thick)

    # 保存
    cv2.imencode('.png', img)[1].tofile(save_path)
    print(f"✅ 已生成: {os.path.basename(save_path)}")

if __name__ == '__main__':
    ORANGE = (0, 165, 255)  
    BLUE = (255, 0, 0)      

    # === 【最新坐标修正】 ===
    my_boxes = [
        # --- 框1 (橙色)：树叶 ---
        {
            'loc': (120, 1, 80, 80),     
            'paste': (10, 310),          
            'color': ORANGE,
            'thickness': 2
        },
        
        # --- 框2 (蓝色)：行人 ---
        {
            'loc': (530, 210, 40, 70),
            'paste': (550, 330),
            'color': BLUE,
            'thickness': 2
        }
    ]

    base_dir = "/home/yelei/shiyan/MMIF-CDDFuse-main"
    img_name = "00706N.png" 

    # === 替换了 ReCoNet 为 FusionGAN ===
    model_configs = [
        ("Infrared",   "MSRS_test/ir"),
        ("Visible",    "MSRS_test/vi"),
        ("CDDFuse",    "results_baseline"), 
        ("Ours",       "results_hf_pro"),   
        ("DenseFuse",  "others/imagefusion_densefuse-master/results_densefuse"),
        ("DIDFuse",    "others/IVIF-DIDFuse-main/results_didfuse"),
        ("U2Fusion",   "others/U2Fusion-master/results_u2fusion"),
        ("DeFusion",   "others/DecompositionForFusion-main/results_defusion"),
        ("FusionGAN",  "others/FusionGAN-master/results_FusionGAN"), # 修改了这里
        ("SDNet",      "others/SDNet-master/results_sdnet"),
    ]

    output_dir = "paper_figures" 
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 开始处理图片: {img_name}")
    
    for label, rel_path in model_configs:
        src_path = os.path.join(base_dir, rel_path, img_name)
        
        # 容错检查
        if not os.path.exists(src_path):
            if label in ["Infrared", "Visible"]:
                alt_path = src_path.replace("/ir/", "/Ir/").replace("/vi/", "/Vi/")
                if os.path.exists(alt_path):
                    src_path = alt_path
                else: continue
            else: 
                print(f"⚠️ 找不到路径跳过: {src_path}") # 增加了一条提示，方便你检查哪个路径写错了
                continue
            
        save_name = f"{label}_{img_name}"
        save_path = os.path.join(output_dir, save_name)
        draw_zoom_box(src_path, save_path, my_boxes)

    print("✨ 全部完成！")