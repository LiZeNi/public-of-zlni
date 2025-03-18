import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embedd_dim, num_heads):
        super(MultiHeadAttention,self).__init__()
        self.embedd_dim=embedd_dim
        self.num_heads=num_heads
        self.head_dim=int(embedd_dim/num_heads)

        self.Q_linner=nn.Linear(embedd_dim,embedd_dim)
        self.K_linear=nn.Linear(embedd_dim,embedd_dim)
        self.V_linaer=nn.Linear(embedd_dim,embedd_dim)
        self.total_linear=nn.Linear(embedd_dim,embedd_dim)
    
    def forward(self,query,key,value,kv_cache=None):
        batch=query.size(0)
        Q=self.Q_linner(query)
        K=self.K_linear(key)
        V=self.V_linaer(value)

        if kv_cache is not None:
            cache_K,cache_V=kv_cache
            K=torch.cat(cache_K,K,dim=1)
            V=torch.cat(cache_V,V,dim=1)
        kv_cache=(K,V)

        Q=Q.view(batch,-1,self.num_heads,self.head_dim).transpose(1,2)
        K=K.view(batch,-1,self.num_heads,self.head_dim).transpose(1,2)
        V=V.view(batch,-1,self.num_heads,self.head_dim).transpose(1,2)

        #计算注意力分数
        scores=torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)
        #计算注意力权重
        weight=F.softmax(scores,dim=-1)
        
        out=torch.matmul(weight,V)
        out=out.transpose(1,2).contiguous().view(batch,-1,self.embedd_dim)
        out=self.total_linear(out)
        return out,weight,kv_cache

if __name__ == "__main__":
    batch=2
    sequence_len=10
    num_heads=8
    embedd_dim=200
    if embedd_dim%num_heads!=0:
        print("请调整embedd_dim和num_heads")

    #创建随机矩阵
    query=torch.rand(size=(batch,sequence_len,embedd_dim))
    key=torch.rand(size=(batch,sequence_len,embedd_dim))
    value=torch.rand(size=(batch,sequence_len,embedd_dim))

    model=MultiHeadAttention(embedd_dim,num_heads)
    out,weight,kv_cache=model(query,key,value)
    print("weight:{}".format(weight))




        


    
