import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeDynamicAggregation(nn.Module):

    def __init__(self,
                 in_channels=2048,
                 alpha=1.0,
                 init_cfg=None
                 ):
        super().__init__()
        self.fc_cls = nn.Linear(in_channels, in_channels//2)
        self.bn1 = nn.BatchNorm1d(in_channels//2)        # 批归一化层
        self.fc2 = nn.Linear(in_channels//2, 1) # 第二个全连接层
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.alpha = alpha
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer1 = nn.Linear(in_channels*2, in_channels//4)
        self.fc_layer2 = nn.Linear(in_channels//4, 2)       
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,labels=None):
        x_mean = x.mean(3).mean(2)
        x_mean_1 = x_mean.view(x_mean.size(0), -1, 1, 1).expand_as(x)
        sim = torch.cosine_similarity(x, x_mean_1, dim=1).unsqueeze(1)
        x_new = (x * sim.expand_as(x)).mean(3).mean(2)
        x = self.alpha * x_new + x_mean
        if labels is None:
            return x
        
        out_feats = []
        label = torch.unique(labels, sorted=False)
        for cls_id in label:
            out_feat = x[labels == cls_id]
            out_feat = torch.mean(out_feat, dim=0).unsqueeze(0)
            # out_feat = torch.mean(out_feat * weight.expand_as(out_feat), dim=0).unsqueeze(0)
            out_feats.append(out_feat)
        proto = torch.cat(out_feats, dim=0)
        return proto

    def attention_forward(self, x):
        x = self.fc_layer1(x)
        x = self.gelu(x)
        x = self.fc_layer2(x)
        return x


