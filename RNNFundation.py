import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2   
num_layer = 1   # 多层RNN

cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layer=num_layer)

inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layer,batch_size,hidden_size)   

out, hidden = cell(inputs, hidden)

print('Output size:', out.shape)
print('Output:', out)
print('Hidden size:', hidden.shape)
print('Hidden:', hidden)

#如何将字符串转为向量矩阵，One-Hot独热编码


#对于输入输出维度的感受