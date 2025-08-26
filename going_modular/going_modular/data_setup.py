"""
包含为图像分类数据创建PyTorch DataLoader的功能。
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """创建训练和测试DataLoader。

  接受训练目录和测试目录路径，将它们转换为PyTorch数据集，然后转换为PyTorch DataLoader。

  参数:
    train_dir: 训练目录的路径。
    test_dir: 测试目录的路径。
    transform: 对训练和测试数据执行的torchvision变换。
    batch_size: 每个DataLoader中每批次的样本数量。
    num_workers: 每个DataLoader的工作进程数量的整数。

  返回:
    (train_dataloader, test_dataloader, class_names)的元组。
    其中class_names是目标类别的列表。
    使用示例:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # 使用ImageFolder创建数据集
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # 获取类别名称
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
