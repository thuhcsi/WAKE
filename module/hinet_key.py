import torch
import sys
sys.path.append("..")
from module.invblock_key import INV_block_key


class Hinet(torch.nn.Module):

    def __init__(self, in_channel=2, num_layers=16):
        super(Hinet, self).__init__()
        self.inv_blocks = torch.nn.ModuleList([INV_block_key(in_channel) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x1, x2,key, rev=False):
        # x1:cover
        # x2:secret
        # print(key.shape)
        if not rev:
            for i,inv_block in enumerate(self.inv_blocks):
                x1, x2 = inv_block(x1, x2,key[i])
        else:
            for i,inv_block in enumerate(reversed(self.inv_blocks)):
                x1, x2 = inv_block(x1, x2, key[self.num_layers-1-i],rev=True)
        return x1, x2
