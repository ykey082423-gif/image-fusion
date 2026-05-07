import torch
import torch.nn as nn
import torch.nn.functional as F
from net_cbam import TransformerBlock

# =============================================================================
# 新增：光照感知模块 (IAM) - 用于识别环境光照质量
# =============================================================================
class IAM(nn.Module):
    def __init__(self, in_channels=1):
        super(IAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid() # 输出权重 w 在 0-1 之间
        )

    def forward(self, x):
        b = x.size(0)
        out = self.conv(x).view(b, -1)
        w = self.fc(out)
        return w.view(b, 1, 1, 1)

# =============================================================================
# 修改后的解码器：支持动态自适应高频注入
# =============================================================================
class Restormer_Decoder_HF_Pro(nn.Module):
    def __init__(self, 
                 dim=64, 
                 out_channels=1, 
                 num_blocks=[1, 1, 1], 
                 heads=[8, 8, 8], 
                 ffn_expansion_factor=2, 
                 bias=False, 
                 LayerNorm_type='WithBias'):
        super(Restormer_Decoder_HF_Pro, self).__init__()
        
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),)
        
        self.sigmoid = nn.Sigmoid()
        self.smooth = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        # --- 创新点：新增光照感知子网络 ---
        self.iam = IAM(in_channels=1)

    def forward(self, inp_img, ir_img, base_feature, detail_feature):
        # 1. 基础特征解码
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out = self.output(out_enc_level1)
        
        # 2. 动态感知环境权重 w
        w = self.iam(inp_img) 
        
        # 3. 物理特征剥离：红外高频
        ir_low = self.smooth(ir_img)
        ir_high = ir_img - ir_low
        
        # 4. 动态非对称注入公式 (创新点：w 联动控制)
        # 解释：w 越大(白天)，越依赖可见光亮度；w 越小(黑夜)，红外注入权重 (2-w) 越大。
        out = self.sigmoid(out + w * inp_img + (2.0 - w) * ir_high)
        
        return out, w