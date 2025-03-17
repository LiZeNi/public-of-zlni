import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self,embedd_dim,num_heads):
        super().__init__()
        self.embedd_dim=embedd_dim
        self.num_heads=num_heads
        self.head_dim=int(embedd_dim/num_heads)

        #定义线性变换
        self.Q_linear=nn.Linear(embedd_dim,embedd_dim)
        self.K_linear=nn.Linear(embedd_dim,embedd_dim)
        self.V_linear=nn.Linear(embedd_dim,embedd_dim)
        self.total_linear=nn.Linear(embedd_dim,embedd_dim)

    def forward(self,query,key,value):
        batch,sequence_len,_=query.size()

        Q=self.Q_linear(query).view(batch,-1,self.num_heads,self.head_dim).transpose(1,2)
        K=self.K_linear(key).view(batch,-1,1,self.head_dim).transpose(1,2).repeat(1,self.num_heads,1,1)
        V=self.V_linear(value).view(batch,-1,1,self.head_dim).transpose(1,2).repeat(1,self.num_heads,1,1)

        #得到注意力分数
        score=torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(self.head_dim)
        #得到注意力权重
        weight=F.softmax(score,dim=1)

        out=torch.matmul(weight,V).transpose(1,2).contiguous().view(batch,sequence_len,self.embedd_dim)
        out=self.total_linear(out)
        return out,weight
if  __name__=="__main__":
    batch=2
    sequence_len=10
    num_heads=8
    embedd_dim=200

    if (embedd_dim%num_heads)!=0:
        print("请调整embedd_dim和num_heads")

    #定义随机矩阵
    query=torch.rand(size=(batch,sequence_len,embedd_dim))
    key=torch.rand(size=(batch,sequence_len,embedd_dim))
    value=torch.rand(size=(batch,sequence_len,embedd_dim))

    model=MultiQueryAttention(embedd_dim,num_heads)
    out,weight=model(query,key,value)
    print("weight{}".format(weight))



