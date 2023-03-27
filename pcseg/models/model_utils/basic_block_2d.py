import torch.nn as nn


class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


def build_block(name, in_channels, out_channels, act_fn=nn.ReLU, norm_layer=nn.BatchNorm2d, **kwargs):
    if name == 'BasicBlock2D':
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    elif name == 'DeConv2dBlock':
        block = [
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    else:
        raise NotImplementedError

    return block
