from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

dataset = ImageFolder(root='datasets/mood', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #shuffle=True表示打乱数据集，batch_size=32表示每个批次32个样本。

for image, label in dataloader:
    print(image.shape)
    print(label.shape)
    break
