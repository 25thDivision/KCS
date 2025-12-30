import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Mamba Block for Mamba-based Models
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2) 
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=True, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4/Mamba Parameter Init
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(self.d_inner, dim=-1)
        
        x_in = x_in.permute(0, 2, 1)
        x_in = self.conv1d(x_in)[:, :, :seq_len] 
        x_in = self.act(x_in)
        
        y = self.ssm(x_in)
        
        y = y * self.act(res.permute(0, 2, 1))
        y = y.permute(0, 2, 1)
        
        return self.out_proj(y)

    def ssm(self, x):
        B, D, L = x.shape
        x_t = x.permute(0, 2, 1)
        x_dbl = self.x_proj(x_t)
        (delta, B_ssm, C_ssm) = x_dbl.split([self.dt_rank, 16, 16], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        D_ssm = self.D.float()
        
        ys = []
        h = torch.zeros(B, D, 16, device=x.device)
        
        for t in range(L):
            dt = delta[:, t, :]
            dA = torch.exp(torch.einsum('bd,dn->bdn', dt, A))
            dB = torch.einsum('bd,bn->bdn', dt, B_ssm[:, t, :])
            x_val = x[:, :, t].unsqueeze(-1)
            h = h * dA + dB * x_val
            y_t = torch.einsum('bdn,bn->bd', h, C_ssm[:, t, :])
            ys.append(y_t)
            
        y = torch.stack(ys, dim=2)
        return y + x * D_ssm.unsqueeze(-1)


# Mamba-based U-Net Model
class UMamba(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_filters=32):
        super().__init__()
        
        # Level 1
        # CNN Block
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, 3, padding=1), 
            nn.BatchNorm2d(base_filters), 
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1), 
            nn.BatchNorm2d(base_filters), 
            nn.ReLU()
        )
        # Mamba Block for Level 1
        self.enc1_mamba = MambaBlock(d_model=base_filters) 
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Level 2
        # CNN Block
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 3, padding=1), 
            nn.BatchNorm2d(base_filters*2), 
            nn.ReLU()
        )
        # Mamba Block for Level 2
        self.enc2_mamba = MambaBlock(d_model=base_filters*2)
        
        # Bottleneck
        # Use Mamba Block in Bottleneck
        self.bottleneck_mamba = MambaBlock(d_model=base_filters*2)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*3, base_filters, 3, padding=1), # Concat: 32 + 64 = 96
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU()
        )

        self.final = nn.Conv2d(base_filters, out_ch, 1)

    def _forward_mamba(self, x, mamba_layer):
        """Helper to process 2D images with Mamba"""
        B, C, H, W = x.shape
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        x_out = mamba_layer(x_seq)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out

    def forward(self, x):
        # [Dynamic Padding]
        h, w = x.shape[2], x.shape[3]
        target_h = ((h - 1) // 2 + 1) * 2 
        target_w = ((w - 1) // 2 + 1) * 2
        pad_h, pad_w = target_h - h, target_w - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Level 1
        x1 = self.enc1_conv(x)
        # Add global information from Mamba to CNN features (like a residual connection)
        x1_mamba = self._forward_mamba(x1, self.enc1_mamba)
        x1 = x1 + x1_mamba 
        
        p1 = self.pool(x1)
        
        # Level 2
        x2 = self.enc2_conv(p1)
        x2_mamba = self._forward_mamba(x2, self.enc2_mamba)
        x2 = x2 + x2_mamba
        
        # Bottleneck
        x_bottle = self._forward_mamba(x2, self.bottleneck_mamba)
        
        # Decoder
        x_up = self.up(x_bottle)
        
        if x_up.shape[2:] != x1.shape[2:]:
            x_up = F.interpolate(x_up, size=x1.shape[2:])
            
        x_cat = torch.cat([x_up, x1], dim=1)
        x_dec = self.dec1(x_cat)
        
        out = self.final(x_dec)

        # Un-pad
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            
        return out


# class UMamba(nn.Module):
#     def __init__(self, in_ch=2, out_ch=2, base_filters=32):
#         super().__init__()
        
#         # Level 1
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(in_ch, base_filters, 3, padding=1), nn.BatchNorm2d(base_filters), nn.ReLU(),
#             nn.Conv2d(base_filters, base_filters, 3, padding=1), nn.BatchNorm2d(base_filters), nn.ReLU()
#         )
#         self.pool = nn.MaxPool2d(2)
        
#         # Level 2 (Encoder part)
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(base_filters, base_filters*2, 3, padding=1), nn.BatchNorm2d(base_filters*2), nn.ReLU()
#         )
        
#         # Bottleneck (Mamba)
#         self.mamba_dim = base_filters * 2
#         self.mamba = MambaBlock(d_model=self.mamba_dim)
        
#         # Decoder
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(base_filters*3, base_filters, 3, padding=1),
#             nn.BatchNorm2d(base_filters),
#             nn.ReLU(),
#             nn.Conv2d(base_filters, base_filters, 3, padding=1),
#             nn.BatchNorm2d(base_filters),
#             nn.ReLU()
#         )

#         self.final = nn.Conv2d(base_filters, out_ch, 1)

#     def forward(self, x):
#         # [Common] Dynamic Padding
#         h, w = x.shape[2], x.shape[3]
#         target_h = ((h - 1) // 2 + 1) * 2 
#         target_w = ((w - 1) // 2 + 1) * 2
#         pad_h, pad_w = target_h - h, target_w - w
        
#         if pad_h > 0 or pad_w > 0:
#             x = F.pad(x, (0, pad_w, 0, pad_h))

#         # Forward
#         x1 = self.enc1(x)
#         p1 = self.pool(x1)
        
#         x2 = self.enc2(p1)
        
#         # Flatten for Mamba
#         B, C, H, W = x2.shape
#         x_seq = x2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
#         x_mamba = self.mamba(x_seq)
        
#         x_mamba = x_mamba.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
#         x_up = self.up(x_mamba)
        
#         if x_up.shape[2:] != x1.shape[2:]:
#             x_up = F.interpolate(x_up, size=x1.shape[2:])
            
#         x_cat = torch.cat([x_up, x1], dim=1)
#         x_dec = self.dec1(x_cat)
        
#         out = self.final(x_dec)

#         # Un-pad
#         if pad_h > 0 or pad_w > 0:
#             out = out[:, :, :h, :w]
            
#         return out