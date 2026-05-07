import os
import cv2
import random
import torch
import numpy as np
import torch.utils.data as Data

class M3FD_Dataset(Data.Dataset):
    def __init__(self, vis_dir, ir_dir, patch_size=256):
        super(M3FD_Dataset, self).__init__()
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.patch_size = patch_size
        
        # 获取所有图片文件名，并确保排序一致
        self.img_names = sorted(os.listdir(vis_dir))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)

        # 读取灰度图
        vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        # 归一化到 0~1
        vis_img = vis_img.astype(np.float32) / 255.0
        ir_img = ir_img.astype(np.float32) / 255.0

        h, w = vis_img.shape

        # --- 动态随机裁剪 (Dynamic Random Crop) ---
        if h > self.patch_size and w > self.patch_size:
            # 随机生成左上角坐标
            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            
            # 同步裁剪可见光和红外（保证像素绝对对齐）
            vis_patch = vis_img[x:x+self.patch_size, y:y+self.patch_size]
            ir_patch = ir_img[x:x+self.patch_size, y:y+self.patch_size]
        else:
            # 如果原图比 256 小（极少见），就直接 resize
            vis_patch = cv2.resize(vis_img, (self.patch_size, self.patch_size))
            ir_patch = cv2.resize(ir_img, (self.patch_size, self.patch_size))

        # 增加通道维度: (1, H, W)
        vis_patch = np.expand_dims(vis_patch, axis=0)
        ir_patch = np.expand_dims(ir_patch, axis=0)

        return torch.Tensor(vis_patch), torch.Tensor(ir_patch)