#本机使用pytorch，常用的conda环境为DP_study
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        d_model: embedding维度
        num_heads: 注意力头数
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q K V 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        # 1. 线性变换得到 Q K V
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. reshape 成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Q shape: (batch, heads, seq_len, d_k)

        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. softmax
        attention = torch.softmax(scores, dim=-1)

        # 5. 加权 V
        out = torch.matmul(attention, V)

        # 6. 拼接 heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        # 7. 最终线性层
        out = self.fc_out(out)

        return out


d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

model = MultiHeadAttention(d_model, num_heads)

x = torch.randn(batch_size, seq_len, d_model)

out = model(x, x, x)

print(out.shape)