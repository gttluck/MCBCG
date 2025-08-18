import torch
import torch.nn as nn
from einops import rearrange
from model.transformer_utils import *
import pytorch_wavelets as DWT 
import pytorch_wavelets


class DMCA(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, bias=False):
        super(DMCA, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 10)
        self.rescale = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query, Key, Value and output projection layers
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # dynamic weight generation
        self.dynamic_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_heads, kernel_size=1, bias=False),
            nn.Softmax(dim=1)  
        )

        # multi-scale feature branch
        self.multi_scale_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
            ) for _ in range(3)  
        ])

        # channel attention 
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # enhanced position embedding
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # global position embedding
        )

    def forward(self, x, y):
        """
        x: input
        y: reference input
        """
        b, c, h, w = x.shape

        # dynamic weight generation
        weights = self.dynamic_weight(x)  # b, num_heads, 1, 1
        weights = rearrange(weights, 'b head 1 1 -> b head 1 1')

        # Query 和 Key-Value 生成
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)  # 拆分为 Key 和 Value

        # multi-head 
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # normalization
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) * self.rescale

        # apply dynamic weights and compute output
        attn = attn * weights
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_c = self.project_out(out)
        
        out_p = self.pos_emb(x)
        multi_scale_feats = [
            F.interpolate(branch(x), size=(h, w), mode='bilinear', align_corners=False)
            for branch in self.multi_scale_branch
        ]
        multi_scale_feat = sum(multi_scale_feats)  
        multi_scale_feat = self.channel_attention(multi_scale_feat) * multi_scale_feat

        # feature fusion
        out = out_c + out_p + multi_scale_feat
       
        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CRM(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.ssa = nn.Sequential(
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.ssa(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
       

# Intensity Enhancement Layer
class IEM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(SIEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)  
        
        x1, x2 = x.chunk(2, dim=1)  #sg
        x = x1 * x2
        x = self.project_out(x) 
        return x 


# Lightweight Cross Attention
class HV_DCE(nn.Module):
    def __init__(self, dim,num_heads,drop_out_rate=0, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = CRM(dim,drop_out_rate=drop_out_rate) # color enhancement
        self.norm = LayerNorm(dim)
        self.ffn = DMCA(dim, num_heads, bias)
        
    def forward(self, x, y):
        x = self.ffn(self.norm(x),self.norm(y))
        x = x+ self.gdfn(self.norm(x))
        return x
    
class I_DCE(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_DCE, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = SIEL(dim)
        self.ffn = DMCA(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x))  
        return x

