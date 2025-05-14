import glob
import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms

root_dir = "./Dataset/"

# 获取所有类别文件夹路径（自动识别73个类别）
all_class_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
class_names = [os.path.basename(os.path.normpath(d)) for d in all_class_dirs]  # 获取类别名称列表

# 获取所有图片路径及其对应标签
image_paths = []
labels = []
for class_idx, class_dir in enumerate(all_class_dirs):
    # 获取该类别下的所有图片路径（支持常见图片格式）
    class_images = sorted(glob.glob(os.path.join(class_dir, "*.[tT][iI][fF]"))) + \
                   sorted(glob.glob(os.path.join(class_dir, "*.jpg"))) + \
                   sorted(glob.glob(os.path.join(class_dir, "*.jpeg")))
    
    # 添加路径和标签
    image_paths.extend(class_images)
    labels.extend([class_idx] * len(class_images))

# 转换为numpy数组
image_paths = np.array(image_paths)
labels = np.array(labels)

class PollenDataset(data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        
        # 加载图像并转换格式
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return len(self.image_paths)

# 数据预处理（参考网页6的标准图像处理流程）
transform = transforms.Compose([
    transforms.Resize((256, 256)),       # 统一图像尺寸
    transforms.RandomHorizontalFlip(),   # 数据增强
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化参数（参考网页6）
    #                      std=[0.229, 0.224, 0.225])
])

# 创建完整数据集
full_dataset = PollenDataset(image_paths, labels, transform=transform)

# 创建分层划分的数据加载器（参考网页6的数据拆分方法）
from sklearn.model_selection import train_test_split

indices = np.arange(len(full_dataset))
train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 创建子数据集
train_dataset = data.Subset(full_dataset, train_indices)
test_dataset = data.Subset(full_dataset, test_indices)

# 创建数据加载器（参考网页7的DataLoader配置）
batch_size = 16
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,    # 提高数据加载效率
    pin_memory=True   # 加速GPU传输（参考网页6）
)

test_loader = data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 验证信息输出
print(f"总类别数: {len(class_names)}")
print(f"总样本数: {len(full_dataset)}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"类别名称: {class_names}") 