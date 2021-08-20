import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value,time, mask=None, dropout=None):
        # time.Size([128, 100]) 
        # time_gap.Size([128,100,100])
        # query.Size([128, 4, 100, 64]) 
        # scores.Size([128,4,100,100])
        # time_gap=self.compute_gap(time,query, key, value)
        scores = torch.matmul(query,key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
      
        # (第一个时间当成1，指数decay归一化)
        time=time.unsqueeze(1).repeat(1,4,1,1)
        # print(scores.shape,time.shape)
        scores=scores*time
        # scores=scores.mul(time_gap)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
     
        p_attn = F.softmax(scores, dim=-1)
      

        # print(p_attn.shape)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def compute_gap(self,time,key,value,query):
        # time=torch.unsqueeze(time,-1)
        # time=time.repeat(1,1,100)
        time1=copy.deepcopy(time)
        time=time.unsqueeze(-1).repeat(1,1,time1.shape[-1])
        for i in range(time1.shape[0]):
          for j in range(time1.shape[1]):
            for k in range(time1.shape[1]):
              time[i,j,k]=time1[i,j]-time1[i,k]
        time=torch.exp(-(torch.abs(time)/60))
        time=time.unsqueeze(1).repeat(1,4,1,1)
        

        print(time.shape)
        return  time