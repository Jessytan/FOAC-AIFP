import torch
from torch import nn
from torch.nn import functional as F

class ConditionalInformationCouplingModule(nn.Module):
    
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels


        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.bn1 = bn(self.in_channels)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )

            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        self.q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.bn2 = bn(self.in_channels)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

    def forward(self,x,kv_x): 
        kv_x = kv_x.mean(0)
        x_qq = self.forward_single(x,kv_x)
        return x_qq

    def forward_single(self, x, kv_x):

        batch_size = x.size(0)
        kv_x = kv_x.unsqueeze(0)
        v_x = self.v(kv_x).view(1, self.inter_channels, -1)
        v_x = v_x.permute(0, 2, 1)

        q_x = self.q(x).view(batch_size, self.inter_channels, -1)
        q_x = q_x.permute(0, 2, 1)

        k_x = self.k(kv_x).view(1, self.inter_channels, -1)

        f = torch.matmul(q_x, k_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, v_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        kv_x = (self.gap(kv_x).view(kv_x.size(0), -1, 1, 1)).expand_as(x)
        mask = torch.cosine_similarity(x, kv_x, dim=1).unsqueeze(1)
        z = W_y * mask.expand_as(W_y) + x

        return z
    