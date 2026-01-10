import torch
import torch.nn as nn
import torchvision.models as models

# MODEL
class EarlyFusionResNet(nn.Module):
    def __init__(self, in_channels=6, backbone='resnet18', pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.encoder = getattr(models, backbone)(weights=weights)
        
        # Modify First Layer
        original_weights = self.encoder.conv1.weight.data.clone()
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Smart Init
        with torch.no_grad():
            n_repeats = in_channels // 3
            for i in range(n_repeats):
                new_conv1.weight[:, i*3:(i+1)*3] = original_weights / n_repeats
        
        self.encoder.conv1 = new_conv1
        self.encoder.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 2)
        )

    def forward(self, *imgs):
        x = torch.cat(imgs, dim=1) 
        feat = self.encoder(x)
        out = self.head(feat)
        return out[:, 0], out[:, 1]