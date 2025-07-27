# utils/model/prompt_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 논문 구현의 핵심 구성요소 ---

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias=False, act=nn.PReLU()):
        super(CAB, self).__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias)
        ]
        self.body = nn.Sequential(*modules_body)
        self.ca = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res += x
        return res

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, x):
        return self.body(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prompt_channels=0):
        super(UpBlock, self).__init__()
        # Upsample + Conv
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # Feature processing block
        self.conv_block = nn.Sequential(
            CAB(out_channels * 2 + prompt_channels), # Skip + Up-conv + Prompt
            CAB(out_channels * 2 + prompt_channels)
        )
        # Reducer to match skip connection dimension
        self.reduce = nn.Conv2d(out_channels * 2 + prompt_channels, out_channels, kernel_size=1)


    def forward(self, x, skip, prompt=None):
        x = self.up(x)
        
        # Padding for odd dimensions
        if x.shape != skip.shape:
            diffY = skip.shape[2] - x.shape[2]
            diffX = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection and upsampled feature
        x = torch.cat([x, skip], dim=1)

        # Inject prompt if available
        if prompt is not None:
            prompt_spatial = prompt.view(prompt.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, prompt_spatial], dim=1)

        x = self.conv_block(x)
        x = self.reduce(x)
        return x

class PromptUnet(nn.Module):
    def __init__(self, in_chans, out_chans, chans=32, num_pool_layers=4, prompt_embed_dim=0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.prompt_embed_dim = prompt_embed_dim

        # Initial convolution
        self.init_conv = nn.Sequential(
            CAB(in_chans),
            CAB(in_chans),
        )

        # Encoder (Down-sampling path)
        self.down_blocks = nn.ModuleList()
        self.down_cabs = nn.ModuleList()
        
        ch = in_chans
        for i in range(num_pool_layers):
            self.down_cabs.append(CAB(ch))
            self.down_blocks.append(DownBlock(ch, ch * 2))
            ch *= 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            CAB(ch),
            CAB(ch),
        )
        
        # Decoder (Up-sampling path)
        self.up_blocks = nn.ModuleList()
        for i in range(num_pool_layers):
            # prompt_ch for this level is prompt_embed_dim only at the middle level, for example
            # This is a simplification of the paper's multi-level prompt injection
            prompt_ch = self.prompt_embed_dim if i == num_pool_layers // 2 else 0
            
            self.up_blocks.append(UpBlock(ch, ch // 2, prompt_channels=prompt_ch))
            ch //= 2
            
        # Final convolution
        self.final_conv = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, x, prompt=None):
        x = self.init_conv(x)
        
        skips = []
        # Encoder
        for i in range(self.num_pool_layers):
            x = self.down_cabs[i](x)
            skips.append(x)
            x = self.down_blocks[i](x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(self.num_pool_layers):
            skip = skips.pop()
            
            # Inject prompt at the middle level of the decoder
            current_prompt = prompt if i == self.num_pool_layers // 2 else None
            
            x = self.up_blocks[i](x, skip, current_prompt)
            
        return self.final_conv(x)