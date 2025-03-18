
import torch
import torch.nn as nn
import torch.nn.functional as F

#定义MLA
class MultiHeadLatentAttention(nn.Module):
    def __init__(self,batch,embedd_dim,sequence_len,num_heads,head_dim,compress_dim,latent_dim ):
        super().__init__()
        self.batch=batch
        self.embedd_dim=embedd_dim
        self.sequence_len=sequence_len
        self.num_heads=num_heads
        self.head_dim=head_dim
        self.compress_dim=compress_dim#query压缩后的维度
        self.head_latent_dim=int(latent_dim/num_heads)
        self.W_Q=nn.Linear(embedd_dim,embedd_dim)
        self.W_K=nn.Linear(embedd_dim,embedd_dim)
        self.W_V=nn.Linear(embedd_dim,embedd_dim)
        self.W_O=nn.Linear(embedd_dim,embedd_dim)
        #上投影矩阵和下投影矩阵
        self.W_DQ=nn.Linear(embedd_dim,compress_dim)
        self.W_UQ=nn.Linear(compress_dim,head_dim)
        self.W_UK=nn.Linear(latent_dim,embedd_dim)
        self.W_UV=nn.Linear(latent_dim,embedd_dim)
        self.W_DKV=nn.Linear(embedd_dim,latent_dim)
        #定义将解耦后的key压缩来缓存的矩阵
        self.W_DK=nn.Linear(2*embedd_dim,latent_dim)
        #定义ROPE的上投影矩阵
        self.W_QR=nn.Linear(compress_dim,embedd_dim)
        self.W_KR=nn.Linear(latent_dim,embedd_dim)

    #定义旋转矩阵
    def RotatedMatrix(self,sequence_len,embedd_dim,consist):
        #得到频率
        frequence=1.0/(consist**(torch.arange(0,embedd_dim,2)[:embedd_dim//2].float()))
        
        #得到位置
        index=torch.arange(sequence_len)
        
        #得到旋转矩阵
        frequence=torch.outer(index,frequence).float()
        RotatedMatrix=torch.polar(torch.ones_like(frequence),frequence)
        return RotatedMatrix
    
    #使用ROPE
    def ROPE(self,com_query,com_key,new_sequence_len,sequence_len,consist,embedd_dim):
        # 解压缩成(batch,sequence_len,embedd_dim)
        query=self.W_QR(com_query)
        key=self.W_KR(com_key)


        #转换成复数
        query=torch.view_as_complex(query.reshape(*query.shape[0:2],-1,2))
        key=torch.view_as_complex(key.reshape(*key.shape[0:2],-1,2))
        
        #旋转
        rotated_query=torch.view_as_real(query*self.RotatedMatrix(sequence_len,embedd_dim,consist))
        retated_key=torch.view_as_real(key*self.RotatedMatrix(new_sequence_len,embedd_dim,consist))
        #(batch,sequence_len,num_heads,head_dim)
        rotated_query=rotated_query.flatten(-2)
        #(batch,new_sequence_len,num_heads,head_dim)
        rotated_key=retated_key.flatten(-2)
        return rotated_query,rotated_key
    
    def forward(self,query,key,value,consist,kv_cache=None):
        
        #(batch,sequence_len,embedd_dim)
        query=self.W_Q(query)
        key=self.W_K(key)
        value=self.W_V(value)
       
       #把之前的key,value提取出来并上投影后拼接，结合上文信息
        if kv_cache!=None:
            cache_k,cache_v=kv_cache#(batch,pre_sequence_len,latent_dim)
            pre_key=self.W_UK(cache_k)
            pre_value=self.W_UV(cache_v)
            key=torch.cat(pre_key,key,dim=1)#(batch,pre_sequence_len+sequence_len,embedd_dim)
            value=torch.cat(pre_value,value,dim=1)
            new_sequence_len=key.size(1)#更新序列长度
        else:
            new_sequence_len=self.sequence_len

        
        #压缩来减少数据量
        com_query=self.W_DQ(query)#(batch,sequence_len,compress_dim)
        com_key=self.W_DKV(key)#(batch,new_sequence_len,latent_dim)
        com_value=self.W_DKV(value)

        #进行ROPE
        rotated_query ,rotated_key=self.ROPE(com_query,com_key,new_sequence_len,self.sequence_len,consist,self.embedd_dim)
        
        #结合位置注意力和信息注意力
        key=torch.cat((key,rotated_key),dim=-1)#(batch,new_sequence_len,2*embedd_dim)
        query=torch.cat((query,rotated_query),dim=-1)#(batch,sequence_len,2*embedd_dim)

        #存储新的上文信息
        cache_k=self.W_DK(key)#(batch,new_sequence,latent_dim)
        cache_v=com_value
        kv_cache=(cache_k,cache_v)

         #分头
        query=query.view(self.batch,-1,self.num_heads,2*self.head_dim)#(batch,sequence_len,num_heads,2*head_dim)
        key=key.view(self.batch,-1,self.num_heads,2*self.head_dim)#(batch,new_sequence_len,num_heads,2*head_dim)
        value=value.view(self.batch,-1,self.num_heads,self.head_dim)#(batch,new_sequence_len,num_heads,head_dim)
        
        #计算注意力分数
        key=key.transpose(1,2)#(batch,,num_heads,new_sequence_len,2*head_dim)
        query=query.transpose(1,2)#(batch,,num_heads,sequence_len,2*head_dim)
        value=value.transpose(1,2)#(batch,,num_heads,new_sequence_len,head_dim)
        scores=torch.matmul(query,key.transpose(-2,-1))/(self.head_dim**0.5)#(batch,,num_heads,sequence_len,new_sequence_len)
    
        #计算注意力权重
        weight=F.softmax(scores,dim=-1)

        out=torch.matmul(weight,value)
        out=out.transpose(1,2).contiguous().view(self.batch,-1,self.embedd_dim)
        out=self.W_O(out)
        return out,weight,kv_cache
    

if __name__=="__main__":
    batch=2
    sequence_len=10
    embedd_dim=200
    num_heads=4
    latent_dim=30
    compress_dim=50
    head_dim=int(embedd_dim/num_heads)
    if embedd_dim%num_heads!=0:
        print("请调整embedd_dim和num_heads")

    #创建随机矩阵
    query=torch.rand(size=(batch,sequence_len,embedd_dim))
    key=torch.rand(size=(batch,sequence_len,embedd_dim))
    value=torch.rand(size=(batch,sequence_len,embedd_dim))

    model=MultiHeadLatentAttention(batch,embedd_dim,sequence_len,num_heads,head_dim,compress_dim,latent_dim )
    out,weight,kv_cache=model(query,key,value,consist=10000.0)
    print("weight:{}".format(weight))


        



