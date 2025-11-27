# -*- coding: utf-8 -*-
"""
数据集类

用于双分支CNN训练的PyTorch数据集
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import pickle
from pathlib import Path


class EEGFeatureDataset(Dataset):
    """
    EEG特征数据集
    
    加载子窗口特征张量用于训练
    """
    
    def __init__(self,
                 features: np.ndarray,
                 labels: np.ndarray,
                 subject_ids: Optional[np.ndarray] = None,
                 transform=None):
        """
        Args:
            features: 特征张量 (n_samples, T, R, F)
            labels: 标签 (n_samples,)
            subject_ids: 受试者ID (可选)
            transform: 数据变换
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        self.transform = transform
        
        self.n_samples = len(labels)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def get_class_weights(self, n_classes: int = 3) -> torch.Tensor:
        """
        计算类别权重（用于处理类别不平衡）
        
        Args:
            n_classes: 总类别数
        """
        labels_np = self.labels.numpy()
        
        # 确保bincount返回所有类别的计数
        class_counts = np.bincount(labels_np, minlength=n_classes)
        
        # 避免除以0
        class_counts = np.maximum(class_counts, 1)
        
        n_samples = len(labels_np)
        weights = n_samples / (n_classes * class_counts)
        
        return torch.FloatTensor(weights)


class EEGRawDataset(Dataset):
    """
    EEG原始数据数据集
    
    从原始EEG数据动态提取特征
    """
    
    def __init__(self,
                 eeg_data: np.ndarray,
                 labels: np.ndarray,
                 channel_names: List[str],
                 feature_extractor,
                 subject_ids: Optional[np.ndarray] = None):
        """
        Args:
            eeg_data: EEG数据 (n_samples, channels, samples)
            labels: 标签
            channel_names: 通道名称
            feature_extractor: 特征提取器实例
            subject_ids: 受试者ID
        """
        self.eeg_data = eeg_data
        self.labels = torch.LongTensor(labels)
        self.channel_names = channel_names
        self.feature_extractor = feature_extractor
        self.subject_ids = subject_ids
        
        self.n_samples = len(labels)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 动态提取特征
        eeg = self.eeg_data[idx]
        features = self.feature_extractor.extract_all_features(eeg, self.channel_names)
        
        x = torch.FloatTensor(features)
        y = self.labels[idx]
        
        return x, y


def create_dataloaders(features: np.ndarray,
                       labels: np.ndarray,
                       subject_ids: np.ndarray,
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42,
                       num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    按受试者划分，避免数据泄露
    
    Args:
        features: 特征张量 (n_samples, T, R, F)
        labels: 标签
        subject_ids: 受试者ID
        batch_size: 批大小
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        num_workers: 数据加载线程数
        
    Returns:
        包含 train/val/test DataLoader 的字典
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_state)
    
    # 获取唯一受试者及其标签
    unique_subjects = np.unique(subject_ids)
    subject_labels = []
    for subj in unique_subjects:
        mask = subject_ids == subj
        subject_labels.append(labels[mask][0])
    subject_labels = np.array(subject_labels)
    
    n_subjects = len(unique_subjects)
    n_classes = len(np.unique(subject_labels))
    
    # 检查受试者数量是否足够进行分层划分
    min_subjects_needed = n_classes * 3  # 每个类至少需要3个受试者
    
    if n_subjects < min_subjects_needed:
        # 受试者太少，使用简单划分（不分层）
        print(f"警告: 受试者数量({n_subjects})较少，使用简单划分而非分层划分")
        
        # 随机打乱
        indices = np.random.permutation(n_subjects)
        
        # 计算划分点
        n_test = max(n_classes, int(n_subjects * test_size))
        n_val = max(1, int(n_subjects * val_size))
        n_train = n_subjects - n_test - n_val
        
        if n_train < 1:
            # 极端情况：受试者太少
            n_train = max(1, n_subjects - 2)
            n_val = 1
            n_test = n_subjects - n_train - n_val
        
        train_subjects = unique_subjects[indices[:n_train]]
        val_subjects = unique_subjects[indices[n_train:n_train+n_val]]
        test_subjects = unique_subjects[indices[n_train+n_val:]]
    else:
        # 正常分层划分
        # 按受试者划分
        train_val_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=test_size,
            stratify=subject_labels,
            random_state=random_state
        )
        
        # 获取训练+验证集标签
        train_val_labels = []
        for subj in train_val_subjects:
            mask = subject_ids == subj
            train_val_labels.append(labels[mask][0])
        train_val_labels = np.array(train_val_labels)
        
        # 划分训练和验证
        val_ratio = val_size / (1 - test_size)
        
        # 检查是否可以进行分层划分
        if len(train_val_subjects) >= n_classes * 2:
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=val_ratio,
                stratify=train_val_labels,
                random_state=random_state
            )
        else:
            # 简单划分
            n_val = max(1, int(len(train_val_subjects) * val_ratio))
            train_subjects = train_val_subjects[:-n_val]
            val_subjects = train_val_subjects[-n_val:]
    
    # 获取样本索引
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # 创建数据集
    train_dataset = EEGFeatureDataset(
        features[train_mask],
        labels[train_mask],
        subject_ids[train_mask]
    )
    
    val_dataset = EEGFeatureDataset(
        features[val_mask],
        labels[val_mask],
        subject_ids[val_mask]
    )
    
    test_dataset = EEGFeatureDataset(
        features[test_mask],
        labels[test_mask],
        subject_ids[test_mask]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本 ({len(train_subjects)} 受试者)")
    print(f"  验证集: {len(val_dataset)} 样本 ({len(val_subjects)} 受试者)")
    print(f"  测试集: {len(test_dataset)} 样本 ({len(test_subjects)} 受试者)")
    
    # 类别分布
    print(f"\n类别分布:")
    for split_name, dataset in [('训练', train_dataset), ('验证', val_dataset), ('测试', test_dataset)]:
        labels_np = dataset.labels.numpy()
        counts = np.bincount(labels_np, minlength=3)
        print(f"  {split_name}: AD={counts[0]}, FTD={counts[1]}, CN={counts[2]}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_weights': train_dataset.get_class_weights()
    }


def load_features_from_pkl(pkl_path: str) -> Dict:
    """
    从pickle文件加载特征
    
    Args:
        pkl_path: pickle文件路径
        
    Returns:
        包含特征、标签等的字典
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("数据集类测试")
    print("=" * 60)
    
    # 创建测试数据
    n_samples = 100
    n_timesteps = 8
    n_regions = 4
    n_features = 26
    n_classes = 3
    
    # 模拟数据
    np.random.seed(42)
    features = np.random.randn(n_samples, n_timesteps, n_regions, n_features).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)
    subject_ids = np.array([f'sub-{i//10:03d}' for i in range(n_samples)])
    
    print(f"\n模拟数据形状: {features.shape}")
    print(f"标签: {labels[:10]}...")
    print(f"受试者: {np.unique(subject_ids)}")
    
    # 创建数据加载器
    dataloaders = create_dataloaders(
        features=features,
        labels=labels,
        subject_ids=subject_ids,
        batch_size=16
    )
    
    # 测试迭代
    print(f"\n测试数据迭代...")
    for batch_x, batch_y in dataloaders['train']:
        print(f"  Batch X shape: {batch_x.shape}")
        print(f"  Batch Y shape: {batch_y.shape}")
        break
    
    print(f"\n类别权重: {dataloaders['class_weights']}")
    
    print("\n测试完成！")

