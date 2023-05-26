import torch.nn as nn
import torch.nn.functional as F
import torch
import config as default_config


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, drop_out=0.1, name=None, config=default_config):
        super(FeatureProjector, self).__init__()
        self.name = name
        self.device = config.DEVICE

        self.feed_foward_size = int(output_dim / 2)
        self.project_size = output_dim - self.feed_foward_size
        self.proj1 = nn.Linear(input_dim, self.feed_foward_size, bias=True)

        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias=False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias=False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(self.feed_foward_size)
        self.layernorm = nn.LayerNorm(self.project_size)
        self.MLP = nn.Sequential(*self.proj2)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.drop(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x
