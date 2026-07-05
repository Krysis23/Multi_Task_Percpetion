import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels,kernel_size=1)
            for in_ch in in_channels_list
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self,features:list) -> list:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs,features)]
        fused = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            deeper = fused[0]
            shallower = laterals[i]

            upsampled = F.interpolate(deeper,size=shallower.shape[-2:], mode="nearest")
            fused_level = shallower + upsampled
            fused.insert(0,fused_level)

        outputs = [conv(f) for conv, f in zip(self.output_convs,fused)]

        return outputs

