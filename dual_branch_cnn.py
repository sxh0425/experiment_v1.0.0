# -*- coding: utf-8 -*-
"""
双分支卷积神经网络模型

架构:
1. 输入: 特征张量 (batch, T, R, F) - 时间步×脑区×特征
2. 时间分支: 沿时间维度卷积，捕捉时序动态
3. 特征分支: 沿特征维度卷积，捕捉特征关系
4. 融合: 支持三种方式 (concat, weighted, cross_attention)
5. 注意力机制 + 全连接层
6. 输出: 3分类 (AD, FTD, CN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import math


class TemporalConvBranch(nn.Module):
    """
    时间卷积分支
    
    沿时间维度进行1D卷积，捕捉特征的时序变化模式
    """
    
    def __init__(self,
                 n_timesteps: int,
                 n_regions: int,
                 n_features: int,
                 hidden_dim: int = 64,
                 kernel_size: int = 3,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            n_timesteps: 时间步数 (T)
            n_regions: 脑区数 (R)
            n_features: 特征数 (F)
            hidden_dim: 隐藏层维度
            kernel_size: 卷积核大小
            n_layers: 卷积层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.n_timesteps = n_timesteps
        self.n_regions = n_regions
        self.n_features = n_features
        self.in_channels = n_regions * n_features
        
        # 卷积层
        layers = []
        in_ch = self.in_channels
        
        for i in range(n_layers):
            out_ch = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.out_channels = in_ch
        
        # 全局平均池化后的全连接层
        self.fc = nn.Linear(self.out_channels, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, R, F)
            
        Returns:
            (batch, hidden_dim)
        """
        batch_size = x.size(0)
        
        # reshape: (batch, T, R, F) -> (batch, R*F, T)
        x = x.view(batch_size, self.n_timesteps, -1)  # (batch, T, R*F)
        x = x.permute(0, 2, 1)  # (batch, R*F, T)
        
        # 卷积
        x = self.conv_layers(x)  # (batch, out_ch, T')
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, out_ch)
        
        # 全连接
        x = self.fc(x)  # (batch, hidden_dim)
        
        return x


class FeatureConvBranch(nn.Module):
    """
    特征卷积分支
    
    沿特征维度进行1D卷积，捕捉不同特征之间的关系
    """
    
    def __init__(self,
                 n_timesteps: int,
                 n_regions: int,
                 n_features: int,
                 hidden_dim: int = 64,
                 kernel_size: int = 3,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            n_timesteps: 时间步数 (T)
            n_regions: 脑区数 (R)
            n_features: 特征数 (F)
            hidden_dim: 隐藏层维度
            kernel_size: 卷积核大小
            n_layers: 卷积层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.n_timesteps = n_timesteps
        self.n_regions = n_regions
        self.n_features = n_features
        self.in_channels = n_timesteps * n_regions
        
        # 卷积层
        layers = []
        in_ch = self.in_channels
        
        for i in range(n_layers):
            out_ch = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        self.out_channels = in_ch
        
        # 全局平均池化后的全连接层
        self.fc = nn.Linear(self.out_channels, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, R, F)
            
        Returns:
            (batch, hidden_dim)
        """
        batch_size = x.size(0)
        
        # reshape: (batch, T, R, F) -> (batch, T*R, F)
        x = x.view(batch_size, -1, self.n_features)  # (batch, T*R, F)
        x = x.permute(0, 1, 2)  # 保持 (batch, T*R, F)
        
        # 转置以便沿F方向卷积: (batch, T*R, F)
        x = x.permute(0, 1, 2)  # (batch, T*R, F)
        
        # 卷积
        x = self.conv_layers(x)  # (batch, out_ch, F')
        
        # 全局平均池化
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, out_ch)
        
        # 全连接
        x = self.fc(x)  # (batch, hidden_dim)
        
        return x


class ConcatFusion(nn.Module):
    """
    方式A: 简单拼接融合
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(dim1 + dim2, output_dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        return self.fc(x)


class WeightedFusion(nn.Module):
    """
    方式B: 加权融合
    
    学习一个权重来平衡两个分支的贡献
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        
        # 权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(dim1 + dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # 投影到相同维度
        self.proj1 = nn.Linear(dim1, output_dim)
        self.proj2 = nn.Linear(dim2, output_dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 计算权重
        concat = torch.cat([x1, x2], dim=1)
        weights = self.weight_net(concat)  # (batch, 2)
        
        # 投影
        p1 = self.proj1(x1)  # (batch, output_dim)
        p2 = self.proj2(x2)  # (batch, output_dim)
        
        # 加权融合
        w1 = weights[:, 0:1]  # (batch, 1)
        w2 = weights[:, 1:2]  # (batch, 1)
        
        return w1 * p1 + w2 * p2


class CrossAttentionFusion(nn.Module):
    """
    方式C: 交叉注意力融合
    
    两个分支相互作为Query和Key/Value
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int, n_heads: int = 4):
        super().__init__()
        
        # 投影到相同维度
        self.d_model = output_dim
        self.proj1 = nn.Linear(dim1, self.d_model)
        self.proj2 = nn.Linear(dim2, self.d_model)
        
        # 多头注意力
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            batch_first=True
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 融合后的处理
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 投影
        p1 = self.proj1(x1).unsqueeze(1)  # (batch, 1, d_model)
        p2 = self.proj2(x2).unsqueeze(1)  # (batch, 1, d_model)
        
        # 交叉注意力
        # x1作为query, x2作为key/value
        attn_1, _ = self.cross_attn_1to2(p1, p2, p2)  # (batch, 1, d_model)
        # x2作为query, x1作为key/value
        attn_2, _ = self.cross_attn_2to1(p2, p1, p1)  # (batch, 1, d_model)
        
        # 拼接
        attn_1 = attn_1.squeeze(1)
        attn_2 = attn_2.squeeze(1)
        fused = torch.cat([attn_1, attn_2], dim=1)  # (batch, d_model*2)
        
        return self.fc(fused)


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    
    学习每个特征通道的重要性权重
    """
    
    def __init__(self, in_features: int, reduction: int = 4):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        return x * weights


class DualBranchCNN(nn.Module):
    """
    双分支卷积神经网络
    
    主模型类，整合时间分支和特征分支
    """
    
    def __init__(self,
                 n_timesteps: int = 8,
                 n_regions: int = 4,
                 n_features: int = 26,
                 n_classes: int = 3,
                 hidden_dim: int = 64,
                 fusion_type: Literal['concat', 'weighted', 'cross_attention'] = 'concat',
                 kernel_size: int = 3,
                 n_conv_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            n_timesteps: 时间步数 (T)
            n_regions: 脑区数 (R)
            n_features: 特征数 (F)
            n_classes: 分类数
            hidden_dim: 隐藏层维度
            fusion_type: 融合方式 ('concat', 'weighted', 'cross_attention')
            kernel_size: 卷积核大小
            n_conv_layers: 卷积层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.n_timesteps = n_timesteps
        self.n_regions = n_regions
        self.n_features = n_features
        self.fusion_type = fusion_type
        
        # 时间分支
        self.temporal_branch = TemporalConvBranch(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            n_layers=n_conv_layers,
            dropout=dropout
        )
        
        # 特征分支
        self.feature_branch = FeatureConvBranch(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            n_layers=n_conv_layers,
            dropout=dropout
        )
        
        # 融合层
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(hidden_dim, hidden_dim, hidden_dim)
        elif fusion_type == 'weighted':
            self.fusion = WeightedFusion(hidden_dim, hidden_dim, hidden_dim)
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(hidden_dim, hidden_dim, hidden_dim)
        else:
            raise ValueError(f"未知的融合类型: {fusion_type}")
        
        # 注意力机制
        self.attention = ChannelAttention(hidden_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征张量 (batch, T, R, F)
            
        Returns:
            分类logits (batch, n_classes)
        """
        # 两个分支并行处理
        temporal_features = self.temporal_branch(x)  # (batch, hidden_dim)
        feature_features = self.feature_branch(x)    # (batch, hidden_dim)
        
        # 融合
        fused = self.fusion(temporal_features, feature_features)  # (batch, hidden_dim)
        
        # 注意力
        attended = self.attention(fused)  # (batch, hidden_dim)
        
        # 分类
        logits = self.classifier(attended)  # (batch, n_classes)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取融合后的特征向量（用于分析）
        """
        temporal_features = self.temporal_branch(x)
        feature_features = self.feature_branch(x)
        fused = self.fusion(temporal_features, feature_features)
        attended = self.attention(fused)
        return attended


class DualBranchCNNWithGraphPlaceholder(DualBranchCNN):
    """
    带有图卷积占位符的双分支CNN
    
    预留图卷积分支的接口，方便未来扩展
    """
    
    def __init__(self,
                 n_timesteps: int = 8,
                 n_regions: int = 4,
                 n_features: int = 26,
                 n_classes: int = 3,
                 hidden_dim: int = 64,
                 fusion_type: Literal['concat', 'weighted', 'cross_attention'] = 'concat',
                 use_graph_branch: bool = False,
                 **kwargs):
        
        super().__init__(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            fusion_type=fusion_type,
            **kwargs
        )
        
        self.use_graph_branch = use_graph_branch
        
        if use_graph_branch:
            # TODO: 在这里添加图卷积分支
            # self.graph_branch = GraphConvBranch(...)
            # 需要修改融合层以接受三个输入
            pass
    
    def forward(self, x: torch.Tensor, 
                adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 特征张量 (batch, T, R, F)
            adj_matrix: 邻接矩阵 (可选，用于图卷积)
            
        Returns:
            分类logits
        """
        if self.use_graph_branch and adj_matrix is not None:
            # TODO: 添加图卷积分支的处理
            pass
        
        return super().forward(x)


def create_model(n_timesteps: int = 8,
                 n_regions: int = 4,
                 n_features: int = 26,
                 n_classes: int = 3,
                 hidden_dim: int = 64,
                 fusion_type: str = 'concat',
                 **kwargs) -> DualBranchCNN:
    """
    创建模型的工厂函数
    
    Args:
        n_timesteps: 时间步数
        n_regions: 脑区数
        n_features: 特征数
        n_classes: 分类数
        hidden_dim: 隐藏层维度
        fusion_type: 融合方式
        
    Returns:
        DualBranchCNN模型
    """
    return DualBranchCNN(
        n_timesteps=n_timesteps,
        n_regions=n_regions,
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        fusion_type=fusion_type,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("双分支CNN模型测试")
    print("=" * 60)
    
    # 模型参数
    batch_size = 4
    n_timesteps = 8
    n_regions = 4
    n_features = 26
    n_classes = 3
    
    # 创建测试输入
    x = torch.randn(batch_size, n_timesteps, n_regions, n_features)
    print(f"\n输入形状: {x.shape}")
    
    # 测试三种融合方式
    for fusion_type in ['concat', 'weighted', 'cross_attention']:
        print(f"\n{'='*40}")
        print(f"测试融合方式: {fusion_type}")
        print('='*40)
        
        model = create_model(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=64,
            fusion_type=fusion_type
        )
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
        
        print(f"  输出形状: {logits.shape}")
        print(f"  预测概率示例:\n{probs[0].numpy()}")
        
        # 获取特征
        features = model.get_features(x)
        print(f"  特征向量形状: {features.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

