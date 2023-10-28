import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96):
        super(mlp, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.hidden = nn.Linear(num_tokens, 4*num_tokens)
        self.head = nn.Linear(4*num_tokens, num_outputs)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = self.head(x)
        return x
