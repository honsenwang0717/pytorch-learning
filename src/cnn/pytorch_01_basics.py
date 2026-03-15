import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
import numpy as np

# 基础使用
w = torch.randn(5,8, requires_grad=True)
b = torch.randn(8)
X = torch.randn(1,5)
Y = torch.randn(1,8)

# Z = torch.randn(2,5,4)
# print(Z)
# print(Z.shape)
# print(Z.size())

y = F.relu(X @ w + b)

print(y.shape)

loss = F.cross_entropy(y,Y)
print(loss)
loss.backward()  #反向传播：把所有的参数（w,b）的梯度计算出来，梯度会自动累加到参数的grad属性中  (要计算梯度，必须设置要进行梯度更新的参数requires_grad=True)
print(w.grad)
print(b.grad)

lr = 0.01
w = w - lr * w.grad # 使用梯度下降法更新参数