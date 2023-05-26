import torch.nn as nn
import torch
import config as default_config
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out=0.3, name=None):
        super(BaseClassifier, self).__init__()
        self.name = name
        # ModuleList = [nn.Dropout(p=drop_out)]
        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)

    def forward(self, x):
        x = self.MLP(x)
        return x
