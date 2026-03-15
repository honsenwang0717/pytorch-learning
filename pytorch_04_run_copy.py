import torch
import torch.nn as nn
from torch import optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pytorch_02_model import Model_CNN
from pytorch_03_data import dataloader, dataset

# 设置 matplotlib 中文显示（可选，没有中文字体可注释掉）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

#(1)检查GPU使用情况
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#(2)创建模型，并移动到GPU
model_cnn = Model_CNN().to(device)
#(3)定义损失函数
loss_function = nn.CrossEntropyLoss()
#(4)定义优化器
opt = optim.Adam(model_cnn.parameters(), lr=0.001)

# 用于画图的记录
loss_history = []          # 每个 batch 的 loss
epoch_losses = []          # 每个 epoch 的平均 loss

#(5)训练模型
epochs = 8
for epoch in range(epochs):
    run_losses = []
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        y = model_cnn(inputs)
        loss = loss_function(y, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_val = loss.item()
        loss_history.append(loss_val)
        run_losses.append(loss_val)
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss_val}")
    epoch_avg = np.mean(run_losses)
    epoch_losses.append(epoch_avg)
    print(f"Epoch {epoch+1} 平均 Loss: {epoch_avg:.4f}")

print("Training finished")

# ========== 用 matplotlib 画图 ==========
# 1. 每个 batch 的 loss 曲线（步数 vs loss）
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='steelblue', alpha=0.7, linewidth=0.5)
plt.xlabel("Step (batch)")
plt.ylabel("Loss")
plt.title("Loss 曲线（每个 batch）")
plt.grid(True, alpha=0.3)

# 2. 每个 epoch 的平均 loss（更平滑）
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), epoch_losses, 'o-', color='coral', linewidth=2, markersize=8)
plt.xlabel("Epoch")
plt.ylabel("平均 Loss")
plt.title("每个 Epoch 的平均 Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. CNN 看到的样本图像：从 dataloader 取一个 batch 显示
batch = next(iter(dataloader))
inputs_batch, labels_batch = batch[0], batch[1]
# 数据做过 Normalize(0.5, 0.5)，显示时要反归一化：x*0.5+0.5
n_show = min(8, inputs_batch.size(0))
class_names = dataset.classes
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for idx in range(n_show):
    img = inputs_batch[idx].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
    img = np.clip(img * 0.5 + 0.5, 0, 1)
    axes[idx].imshow(img)
    axes[idx].set_title(f"标签: {class_names[labels_batch[idx].item()]}")
    axes[idx].axis("off")
plt.suptitle("CNN 输入样本（一个 batch 中的前 8 张）")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
plt.show()






