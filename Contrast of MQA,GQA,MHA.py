from GQA import GroupQueryAttention
from MQA import MultiQueryAttention
from MHA import MultiHeadAttention
import torch
#输入相同embedd_dim,sequence_len,num_heads,head_dim,比较输出差值与权重差值
#开始对比三种注意力算法
if __name__=="__main__":
    batch=2
    sequence_len=10
    num_heads=8
    embedd_dim=200
    num_groups=4

    #定义随机矩阵
    query=torch.rand(batch,sequence_len,embedd_dim)
    key=torch.rand(batch,sequence_len,embedd_dim)
    value=torch.rand(batch,sequence_len,embedd_dim)

    model1=GroupQueryAttention(embedd_dim,num_heads,num_groups)
    model2=MultiQueryAttention(embedd_dim,num_heads)
    model3=MultiHeadAttention(embedd_dim,num_heads)
    out1,weight1=model1(query,key,value)
    out2,weight2=model2(query,key,value)
    out3,weight3,kv_cache=model3(query,key,value)
    contrast12=out1-out2
    contrast23=out2-out3
    contrast31=out3-out1
    print("GQA:weight_size{}".format(weight1.size()))
    print("MQA:weight_size{}".format(weight2.size()))
    print("MHA:weight_size{}".format(weight3.size()))
    #将权重维度调到一致
    weight1=weight1.repeat(1,1,1,4)
    weight3=weight3.repeat(1,1,1,8)
    weight12=weight1-weight2
    weight23=weight2-weight3
    weight31=weight3-weight1
    print("GQA-MQA:out{}/n,weight/n{}".format(contrast12,weight12))
    print("MQA-MHA:out{}/n,weight/n{}".format(contrast23,weight23))
    print("MHA-GQA:out{}/n,weight/n{}".format(contrast31,weight31))
    #GQA与MQA权重十分接近且因为GQA分组共享key,value和MQA全部共享key,value,权重差值具有一定规律
    #MQA与MHA权重差距较大，因为两者每个头考虑维度差距较大
    #MHA与GQA权重差距也有点大，但比MQA与MHA小，因为GQA是对MHA的多个head进行分组，且本代码中单个组内只要2个head，所以差距小
    