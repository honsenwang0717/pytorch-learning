import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 生成数据集
#    make_moons 会生成两个交错的半月形分布，是经典的非线性二分类数据
#    noise 控制数据点的散布程度，越大越分散
# ============================================================
X, y = make_moons(n_samples=2000, noise=0.25, random_state=42)

# 把原始数据保存成 CSV，方便查看数据长什么样
# X 有两列特征 (x1, x2)，y 是标签 (0 或 1)
import os
save_dir = os.path.join("datasets", "make_moons")
os.makedirs(save_dir, exist_ok=True)

df = pd.DataFrame(X, columns=["x1", "x2"])
df["label"] = y
save_path = os.path.join(save_dir, "make_moons_dataset.csv")
df.to_csv(save_path, index=False)
print(f"数据集已保存到 {save_path}，共 {len(df)} 条数据")
print(f"前 5 条数据：\n{df.head()}\n")

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转成 PyTorch 张量
# X 是 float32（特征），y 是 float32（因为 BCELoss 要求标签也是浮点数）
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


# ============================================================
# 2. 定义二分类网络
#    输入 2 维特征 → 隐藏层 → 输出 1 个值（经过 Sigmoid 变成概率）
# ============================================================
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # 注意：这里不加 Sigmoid，因为我们用 BCEWithLogitsLoss
            # 它内部自带 Sigmoid，数值上更稳定
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 把输出从 (batch, 1) 压成 (batch,)


# ============================================================
# 3. 构建 DataLoader（小批量训练）
# ============================================================
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# batch_size: 每次喂给网络多少样本
# shuffle=True: 每个 epoch 打乱数据顺序，防止模型记住数据的排列
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ============================================================
# 4. 初始化模型、损失函数、优化器
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BinaryClassifier().to(device)

# BCEWithLogitsLoss = Sigmoid + BCELoss 的合体
# 好处：数值稳定，不会因为 log(0) 而爆炸
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.01,weight_decay=1e-3)

# 学习率调度器：训练过程中逐步降低学习率，帮助后期更精细地收敛
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# ============================================================
# 5. 训练循环
# ============================================================
epochs = 100
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):
    # ---------- 训练阶段 ----------
    model.train()  # 切换到训练模式（影响 Dropout、BatchNorm 等层的行为）
    batch_losses = []
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 前向传播：模型输出 logits（未经 Sigmoid 的原始值）
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        # 反向传播三步曲：清零梯度 → 计算梯度 → 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
  
        # 统计准确率：logits > 0 等价于 Sigmoid(logits) > 0.5，即预测为正类
        preds = (logits > 0).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    scheduler.step()  # 每个 epoch 结束后更新学习率

    epoch_train_loss = np.mean(batch_losses)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # ---------- 验证阶段 ----------
    model.eval()  # 切换到评估模式
    batch_losses_val = []
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # 验证时不需要计算梯度，节省显存和时间
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            batch_losses_val.append(loss.item())

            preds = (logits > 0).float()
            correct_val += (preds == y_batch).sum().item()
            total_val += y_batch.size(0)

    epoch_test_loss = np.mean(batch_losses_val)
    epoch_test_acc = correct_val / total_val
    test_losses.append(epoch_test_loss)
    test_accs.append(epoch_test_acc)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1:3d}/{epochs}]  "
            f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f}  |  "
            f"Test  Loss: {epoch_test_loss:.4f}  Acc: {epoch_test_acc:.4f}"
        )

print("训练完成！")


# ============================================================
# 6. 评估指标（准确率、精确率、召回率、F1）
# ============================================================
model.eval()
with torch.no_grad():
    logits_all = model(X_test.to(device))
    preds_all = (logits_all > 0).float().cpu()

y_true = y_test.numpy()
y_pred = preds_all.numpy()

# TP / FP / FN / TN 是二分类混淆矩阵的四个基本量
TP = ((y_pred == 1) & (y_true == 1)).sum()
FP = ((y_pred == 1) & (y_true == 0)).sum()
FN = ((y_pred == 0) & (y_true == 1)).sum()
TN = ((y_pred == 0) & (y_true == 0)).sum()

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 预测为正的里面有多少真正
recall = TP / (TP + FN) if (TP + FN) > 0 else 0     # 真正为正的里面找回了多少
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n========== 测试集评估 ==========")
print(f"Accuracy  (准确率): {accuracy:.4f}")
print(f"Precision (精确率): {precision:.4f}")
print(f"Recall    (召回率): {recall:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"混淆矩阵: TP={TP}, FP={FP}, FN={FN}, TN={TN}")


# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Loss 曲线
axes[0, 0].plot(train_losses, label="Train Loss", color="steelblue")
axes[0, 0].plot(test_losses, label="Test Loss", color="coral")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss 曲线")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) Accuracy 曲线
axes[0, 1].plot(train_accs, label="Train Acc", color="steelblue")
axes[0, 1].plot(test_accs, label="Test Acc", color="coral")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("准确率曲线")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) 决策边界可视化
#     在特征空间铺一层密集网格，让模型对每个网格点预测，再用颜色填充
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

model.eval()
with torch.no_grad():
    grid_preds = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)

axes[1, 0].contourf(xx, yy, grid_preds, levels=50, cmap="RdYlBu", alpha=0.8)
axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdYlBu",
                    edgecolors="k", s=20, alpha=0.7)
axes[1, 0].set_title("决策边界（测试集）")
axes[1, 0].set_xlabel("x1")
axes[1, 0].set_ylabel("x2")

# (d) 混淆矩阵热力图
conf_matrix = np.array([[TN, FP], [FN, TP]])
im = axes[1, 1].imshow(conf_matrix, cmap="Blues")
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(["预测 0", "预测 1"])
axes[1, 1].set_yticklabels(["真实 0", "真实 1"])
axes[1, 1].set_title("混淆矩阵")
for i in range(2):
    for j in range(2):
        axes[1, 1].text(j, i, str(conf_matrix[i, j]),
                        ha="center", va="center", fontsize=18, color="black")
fig.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("binary_classification_result.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n图片已保存为 binary_classification_result.png")
