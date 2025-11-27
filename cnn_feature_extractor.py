# -*- coding: utf-8 -*-
"""
CNN时频特征提取模块

基于论文: Attention-augmented stockwell transform and convolutional neural network 
          framework for electroencephalogram-based multi-class classification of 
          Frontotemporal Dementia (Xie et al., 2025)

架构:
1. S-Transform 生成时频矩阵
2. 3个CNN块提取空间特征
3. 注意力机制加权特征
4. 全连接层输出特征向量
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from s_transform import StockwellTransform, stockwell_transform_batch


class CNNBlock(nn.Module):
    """
    CNN块: Conv2D -> BatchNorm -> ReLU -> MaxPool
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Tuple[int, int] = (3, 3),
                 pool_size: Tuple[int, int] = (2, 2)):
        """
        初始化CNN块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            pool_size: 池化大小
        """
        super(CNNBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size[0] // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class AttentionBlock(nn.Module):
    """
    注意力机制块
    
    根据论文公式:
    α^l = Softmax(W_att · Z_pooled^l + b_att)
    Z_att^l = α^l · Z_pooled^l
    """
    
    def __init__(self, in_features: int):
        """
        初始化注意力块
        
        Args:
            in_features: 输入特征通道数
        """
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, channels, height, width)
            
        Returns:
            加权后的特征
        """
        # 计算注意力权重
        weights = self.attention(x)  # (batch, channels)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        
        # 应用注意力权重
        return x * weights


class StockwellCNNFeatureExtractor(nn.Module):
    """
    Stockwell-CNN 特征提取器
    
    按照论文架构:
    1. S-Transform 时频表示
    2. 3个CNN块 (32 -> 64 -> 128 filters)
    3. 注意力机制
    4. 全连接层提取特征向量
    """
    
    def __init__(self, 
                 n_freqs: int = 64,
                 n_times: int = 256,
                 n_channels: int = 19,
                 feature_dim: int = 128,
                 dropout: float = 0.5):
        """
        初始化特征提取器
        
        Args:
            n_freqs: 频率维度
            n_times: 时间维度
            n_channels: EEG通道数
            feature_dim: 输出特征维度
            dropout: Dropout率
        """
        super(StockwellCNNFeatureExtractor, self).__init__()
        
        self.n_channels = n_channels
        self.feature_dim = feature_dim
        
        # 输入：每个通道的S-Transform时频图
        # 形状: (batch, n_channels, n_freqs, n_times)
        
        # CNN块 1: 32 filters
        self.cnn_block1 = CNNBlock(n_channels, 32)
        
        # CNN块 2: 64 filters
        self.cnn_block2 = CNNBlock(32, 64)
        
        # CNN块 3: 128 filters
        self.cnn_block3 = CNNBlock(64, 128)
        
        # 注意力机制
        self.attention = AttentionBlock(128)
        
        # 计算展平后的特征维度
        # 假设输入为 (batch, 19, 64, 256)
        # 经过3次2x2池化: 64/8=8, 256/8=32
        self._calculate_flatten_size(n_freqs, n_times)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, feature_dim)
    
    def _calculate_flatten_size(self, n_freqs: int, n_times: int):
        """计算展平后的特征维度"""
        # 模拟前向传播计算维度
        h, w = n_freqs, n_times
        for _ in range(3):  # 3个池化层
            h = h // 2
            w = w // 2
        self.flatten_size = 128 * h * w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: S-Transform时频图 (batch, n_channels, n_freqs, n_times)
            
        Returns:
            特征向量 (batch, feature_dim)
        """
        # CNN块
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        
        # 注意力机制
        x = self.attention(x)
        
        # 全连接层
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征（与forward相同，用于推理）
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class StockwellCNNClassifier(nn.Module):
    """
    完整的 Stockwell-CNN 分类器
    
    用于三分类任务 (AD vs FTD vs CN)
    """
    
    def __init__(self, 
                 n_freqs: int = 64,
                 n_times: int = 256,
                 n_channels: int = 19,
                 n_classes: int = 3,
                 feature_dim: int = 128,
                 dropout: float = 0.5):
        """
        初始化分类器
        
        Args:
            n_freqs: 频率维度
            n_times: 时间维度
            n_channels: EEG通道数
            n_classes: 类别数
            feature_dim: 特征维度
            dropout: Dropout率
        """
        super(StockwellCNNClassifier, self).__init__()
        
        # 特征提取器
        self.feature_extractor = StockwellCNNFeatureExtractor(
            n_freqs=n_freqs,
            n_times=n_times,
            n_channels=n_channels,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: S-Transform时频图
            
        Returns:
            分类logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量
        """
        return self.feature_extractor.extract_features(x)


class MultiChannelStockwellCNN(nn.Module):
    """
    多通道 Stockwell-CNN
    
    每个脑区单独处理，然后融合特征
    """
    
    def __init__(self,
                 n_freqs: int = 64,
                 n_times: int = 256,
                 region_channels: Dict[str, int] = None,
                 feature_dim_per_region: int = 64,
                 n_classes: int = 3,
                 dropout: float = 0.5):
        """
        初始化
        
        Args:
            n_freqs: 频率维度
            n_times: 时间维度
            region_channels: 各脑区通道数
            feature_dim_per_region: 每个脑区的特征维度
            n_classes: 类别数
            dropout: Dropout率
        """
        super(MultiChannelStockwellCNN, self).__init__()
        
        if region_channels is None:
            region_channels = {
                'frontal': 7,
                'temporal': 4,
                'parietal': 3,
                'occipital': 2
            }
        
        self.region_channels = region_channels
        self.regions = list(region_channels.keys())
        
        # 为每个脑区创建特征提取器
        self.region_extractors = nn.ModuleDict()
        for region, n_ch in region_channels.items():
            self.region_extractors[region] = StockwellCNNFeatureExtractor(
                n_freqs=n_freqs,
                n_times=n_times,
                n_channels=n_ch,
                feature_dim=feature_dim_per_region,
                dropout=dropout
            )
        
        # 融合层
        total_features = len(region_channels) * feature_dim_per_region
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_dict: 各脑区的时频图字典 {region_name: tensor}
            
        Returns:
            分类logits
        """
        region_features = []
        
        for region in self.regions:
            if region in x_dict:
                feat = self.region_extractors[region](x_dict[region])
                region_features.append(feat)
        
        # 拼接所有脑区特征
        combined = torch.cat(region_features, dim=1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits
    
    def extract_features(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        提取融合后的特征
        """
        self.eval()
        with torch.no_grad():
            region_features = []
            for region in self.regions:
                if region in x_dict:
                    feat = self.region_extractors[region](x_dict[region])
                    region_features.append(feat)
            
            combined = torch.cat(region_features, dim=1)
            fused = self.fusion(combined)
            
            return fused


def prepare_stockwell_input(eeg_data: np.ndarray,
                            sample_rate: float = 500.0,
                            freq_range: Tuple[float, float] = (0.5, 45.0),
                            n_freqs: int = 64,
                            target_times: int = 256) -> np.ndarray:
    """
    准备Stockwell-CNN的输入
    
    Args:
        eeg_data: EEG数据 (channels x samples) 或 (batch x channels x samples)
        sample_rate: 采样率
        freq_range: 频率范围
        n_freqs: 频率采样点数
        target_times: 目标时间维度
        
    Returns:
        时频图 (batch x channels x n_freqs x target_times) 或 (channels x n_freqs x target_times)
    """
    st = StockwellTransform(sample_rate)
    
    if eeg_data.ndim == 2:
        # 单个样本: (channels, samples)
        n_channels, n_samples = eeg_data.shape
        tf_images = np.zeros((n_channels, n_freqs, n_samples))
        
        for ch in range(n_channels):
            S, freqs, times = st.fast_transform(
                eeg_data[ch], freq_range, n_freqs
            )
            tf_images[ch] = np.abs(S)
        
        # 调整时间维度
        if n_samples != target_times:
            from scipy.ndimage import zoom
            zoom_factor = (1, 1, target_times / n_samples)
            tf_images = zoom(tf_images, zoom_factor, order=1)
        
        return tf_images
    
    elif eeg_data.ndim == 3:
        # 批量样本: (batch, channels, samples)
        batch_size, n_channels, n_samples = eeg_data.shape
        tf_images = np.zeros((batch_size, n_channels, n_freqs, target_times))
        
        for b in range(batch_size):
            tf_images[b] = prepare_stockwell_input(
                eeg_data[b], sample_rate, freq_range, n_freqs, target_times
            )
        
        return tf_images
    
    else:
        raise ValueError(f"不支持的数据维度: {eeg_data.ndim}")


def extract_cnn_features(eeg_data: np.ndarray,
                         model: nn.Module = None,
                         sample_rate: float = 500.0,
                         device: str = 'cpu') -> np.ndarray:
    """
    使用CNN提取时频特征
    
    Args:
        eeg_data: EEG数据 (batch x channels x samples)
        model: CNN模型（如果为None则创建新模型）
        sample_rate: 采样率
        device: 计算设备
        
    Returns:
        特征向量 (batch x feature_dim)
    """
    # 准备输入
    tf_images = prepare_stockwell_input(eeg_data, sample_rate)
    
    # 转换为张量
    tf_tensor = torch.FloatTensor(tf_images)
    
    if tf_tensor.dim() == 3:
        tf_tensor = tf_tensor.unsqueeze(0)  # 添加batch维度
    
    tf_tensor = tf_tensor.to(device)
    
    # 创建或使用模型
    if model is None:
        n_channels = tf_images.shape[-3] if tf_images.ndim == 4 else tf_images.shape[0]
        n_freqs = tf_images.shape[-2]
        n_times = tf_images.shape[-1]
        
        model = StockwellCNNFeatureExtractor(
            n_freqs=n_freqs,
            n_times=n_times,
            n_channels=n_channels
        )
    
    model = model.to(device)
    model.eval()
    
    # 提取特征
    with torch.no_grad():
        features = model(tf_tensor)
    
    return features.cpu().numpy()


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("CNN时频特征提取器测试")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    n_channels = 19
    n_samples = 2000  # 4秒 @ 500Hz
    sample_rate = 500.0
    
    # 模拟EEG数据
    np.random.seed(42)
    eeg_data = np.random.randn(batch_size, n_channels, n_samples).astype(np.float32)
    
    print(f"\n输入数据形状: {eeg_data.shape}")
    
    # 准备S-Transform输入
    print("\n1. 准备S-Transform时频图...")
    tf_images = prepare_stockwell_input(eeg_data, sample_rate)
    print(f"   时频图形状: {tf_images.shape}")
    
    # 创建模型
    print("\n2. 创建Stockwell-CNN模型...")
    model = StockwellCNNFeatureExtractor(
        n_freqs=tf_images.shape[2],
        n_times=tf_images.shape[3],
        n_channels=n_channels,
        feature_dim=128
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    
    # 前向传播测试
    print("\n3. 前向传播测试...")
    tf_tensor = torch.FloatTensor(tf_images)
    
    model.eval()
    with torch.no_grad():
        features = model(tf_tensor)
    
    print(f"   输出特征形状: {features.shape}")
    
    # 测试分类器
    print("\n4. 测试完整分类器...")
    classifier = StockwellCNNClassifier(
        n_freqs=tf_images.shape[2],
        n_times=tf_images.shape[3],
        n_channels=n_channels,
        n_classes=3
    )
    
    with torch.no_grad():
        logits = classifier(tf_tensor)
        probs = F.softmax(logits, dim=1)
    
    print(f"   分类logits形状: {logits.shape}")
    print(f"   预测概率:\n{probs.numpy()}")
    
    # 测试多脑区模型
    print("\n5. 测试多脑区模型...")
    region_data = {
        'frontal': torch.randn(batch_size, 7, 64, 256),
        'temporal': torch.randn(batch_size, 4, 64, 256),
        'parietal': torch.randn(batch_size, 3, 64, 256),
        'occipital': torch.randn(batch_size, 2, 64, 256)
    }
    
    multi_model = MultiChannelStockwellCNN(
        n_freqs=64,
        n_times=256,
        n_classes=3
    )
    
    with torch.no_grad():
        logits = multi_model(region_data)
    
    print(f"   多脑区模型输出形状: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


