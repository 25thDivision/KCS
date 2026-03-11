import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net Model
class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_filters=32):
        super().__init__()
        
        # Level 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU()
        )
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Level 2 (Bottleneck)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU()
        )
        # Bottleneck (CNN)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(),
            nn.Conv2d(base_filters*4, base_filters*2, 3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU()
        )
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*3, base_filters, 3, padding=1), # Concat (32 + 64 = 96)
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(base_filters, out_ch, 1)

    def forward(self, x):
        # Dynamic Padding
        h, w = x.shape[2], x.shape[3]
        target_h = ((h - 1) // 2 + 1) * 2 
        target_w = ((w - 1) // 2 + 1) * 2
        
        pad_h = target_h - h
        pad_w = target_w - w
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        # Forward Pass
        x1 = self.enc1(x)       # Level 1
        p1 = self.pool(x1)
        
        x2 = self.enc2(p1)      # Level 2
        x_bottle = self.bottleneck(x2)
        
        x_up = self.up(x_bottle) # Up
        
        if x_up.shape[2:] != x1.shape[2:]:
            x_up = F.interpolate(x_up, size=x1.shape[2:])
            
        x_cat = torch.cat([x_up, x1], dim=1) # Skip Connection
        x_dec = self.dec1(x_cat)
        
        out = self.final(x_dec)
        
        # Remove Padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            
        return out