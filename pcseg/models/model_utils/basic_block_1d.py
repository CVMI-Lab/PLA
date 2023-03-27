import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(self, channels, norm_fn=None, num_layers=2, last_norm_fn=False, last_bias=True):
        assert len(channels) >= 2
        modules = []
        for i in range(num_layers - 1):
            modules.append(nn.Linear(channels[i], channels[i + 1]))
            if norm_fn:
                modules.append(norm_fn(channels[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(channels[-2], channels[-1], bias=last_bias))
        if last_norm_fn:
            modules.append(norm_fn(channels[-1]))
            modules.append(nn.ReLU())
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        if isinstance(self[-1], nn.Linear):
            nn.init.normal_(self[-1].weight, 0, 0.01)
            nn.init.constant_(self[-1].bias, 0)


def build_block(name, in_channels, out_channels, act_fn=nn.ReLU, norm_layer=nn.BatchNorm1d, **kwargs):
    if name == 'BasicBlock1D':
        block = [
            nn.Linear(in_channels, out_channels),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    elif name == 'DeConv1dBlock':
        block = [
            nn.ConvTranspose1d(in_channels, out_channels, **kwargs),
            norm_layer(out_channels, eps=1e-3, momentum=0.01),
            act_fn()
        ]
    else:
        raise NotImplementedError

    return block
