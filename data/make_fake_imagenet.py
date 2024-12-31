import os
import random
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义伪ImageNet数据集生成函数
def create_fake_imagenet(root_dir, num_classes=10, num_samples_per_class=50, img_size=(224, 224)):
    """
    创建伪ImageNet数据集。
    Args:
        root_dir (str): 数据集存放的根目录。
        num_classes (int): 类别数量。
        num_samples_per_class (int): 每个类别的样本数量。
        img_size (tuple): 生成图片的大小 (宽, 高)。
    """
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建类别目录并生成图片
    for class_idx in range(num_classes):
        class_dir = os.path.join(root_dir, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)
        for sample_idx in range(num_samples_per_class):
            # 生成随机图片 (RGB 图像)
            img_array = np.random.randint(0, 256, (img_size[1], img_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            # 保存图片
            img.save(os.path.join(class_dir, f"img_{sample_idx}.jpg"))

    print(f"伪ImageNet数据集已生成，路径：{root_dir}")


# 生成伪数据集
root_dir = "./fake_imagenet"
create_fake_imagenet(root_dir, num_classes=5, num_samples_per_class=20, img_size=(224, 224))


# 加载伪数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 使用 torchvision 的 ImageFolder
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 打印部分加载结果
print(f"数据集类别: {dataset.classes}")
for images, labels in data_loader:
    print(f"图片批次尺寸: {images.shape}, 标签: {labels}")
    break
