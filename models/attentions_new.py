import torch
from torch import nn
class AEmbed(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(AEmbed,self).__init__()
        self.linear = nn.Linear(in_dim,out_dim)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.relu(self.linear(x))
        return x

class DGAttention(nn.Module):
    def __init__(self,feat_num,in_dim=256):
        super(DGAttention, self).__init__()
        self.feat_num = feat_num
        self.in_dim = in_dim

        for i in range(feat_num):
            setattr(self,"gamma"+str(i),nn.Parameter(torch.zeros(1)))
            setattr(self,"query"+str(i),AEmbed(in_dim,in_dim))
            setattr(self,"key"+str(i),AEmbed(in_dim,in_dim))
            setattr(self,"value"+str(i),AEmbed(in_dim,in_dim))

    def forward(self,*feat_ivs):
        ret_feats,ret_alphas = [],[]
        for i,key in enumerate(feat_ivs):
            Bt,Dimt = key.size()
            pro_key = getattr(self,"key"+str(i))(key).view(Bt,-1,Dimt)
            pro_value = getattr(self,"value"+str(i))(key).view(Bt,-1,Dimt)
            outs,means = [],[]
            for j,query in enumerate(feat_ivs):
                pro_query = getattr(self,"query"+str(i))(query).view(Bt,-1,Dimt).permute(0,2,1)
                energy = torch.bmm(pro_query,pro_key)
                means.append(energy.mean().item())
                attention = torch.softmax(energy,dim=-1)
                out = torch.bmm(pro_value,attention.permute(0,2,1))
                outs.append(out)
            ret_alphas.append(torch.tensor(means).mean())
            feat = torch.stack(outs,dim=0).sum(0).view(Bt,Dimt)
            ret_feats.append((feat/3.0)*getattr(self,"gamma"+str(i))+key)
        ret_alphas = torch.softmax(torch.tensor(ret_alphas),dim=0)
        return ret_feats,ret_alphas