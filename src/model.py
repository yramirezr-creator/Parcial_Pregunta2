"""U-Net propia (sin bibliotecas de segmentación). Entrada: 2 canales (VV, VH). Salida: logits."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base=64, depth=4):
        super().__init__()
        feats = [base * 2 ** i for i in range(depth)]
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        c = in_ch
        for f in feats:
            self.downs.append(DoubleConv(c, f))
            c = f
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2)
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))
        self.head = nn.Conv2d(feats[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-(i // 2 + 1)]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return self.head(x)  # logits (B,1,H,W)


def build_model(in_ch=2, base=64, depth=4, device=None):
    m = UNet(in_ch=in_ch, base=base, depth=depth)
    return m.to(device) if device is not None else m


if __name__ == "__main__":
    m = UNet().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 2, 256, 256))
    print("salida", tuple(y.shape), "| parámetros:", f"{sum(p.numel() for p in m.parameters()):,}")
