# -*- coding: utf-8 -*-
"""
图卷积分支模块

将脑区作为节点，学习脑区间的连接性和信息传递
支持三种邻接矩阵定义方式：
A. 预定义 (predefined) - 基于解剖学连接
B. 可学习 (learnable) - 网络自己学习连接权重
C. 数据驱动 (adaptive) - 基于特征相关性动态计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
import numpy as np


# 预定义的脑区邻接矩阵（基于解剖学连接）
# 脑区顺序: frontal, temporal, parietal, occipital
PREDEFINED_ADJACENCY = np.array([
    [1.0, 0.7, 0.5, 0.3],  # frontal 与其他脑区的连接
    [0.7, 1.0, 0.6, 0.4],  # temporal
    [0.5, 0.6, 1.0, 0.8],  # parietal
    [0.3, 0.4, 0.8, 1.0],  # occipital
], dtype=np.float32)


class GraphConvLayer(nn.Module):
    """
    基础图卷积层
    
    实现: H' = σ(D^(-1/2) A D^(-1/2) H W)
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 (batch, n_nodes, in_features)
            adj: 邻接矩阵 (n_nodes, n_nodes) 或 (batch, n_nodes, n_nodes)
            
        Returns:
            更新后的节点特征 (batch, n_nodes, out_features)
        """
        # 归一化邻接矩阵
        adj_norm = self._normalize_adj(adj)
        
        # 图卷积: A * X * W
        support = torch.matmul(x, self.weight)  # (batch, n_nodes, out_features)
        
        if adj_norm.dim() == 2:
            # 共享邻接矩阵
            output = torch.matmul(adj_norm, support)
        else:
            # 每个样本不同的邻接矩阵
            output = torch.bmm(adj_norm, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """对称归一化邻接矩阵"""
        if adj.dim() == 2:
            # (n_nodes, n_nodes)
            d = adj.sum(dim=1)
            d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
            d_mat = torch.diag(d_inv_sqrt)
            return torch.mm(torch.mm(d_mat, adj), d_mat)
        else:
            # (batch, n_nodes, n_nodes)
            d = adj.sum(dim=2)
            d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
            d_mat = torch.diag_embed(d_inv_sqrt)
            return torch.bmm(torch.bmm(d_mat, adj), d_mat)


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)
    
    动态学习节点间的注意力权重
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int = 4,
                 dropout: float = 0.3,
                 concat: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        
        # 每个头的输出维度
        self.head_dim = out_features // n_heads if concat else out_features
        
        # 线性变换
        self.W = nn.Linear(in_features, self.head_dim * n_heads, bias=False)
        
        # 注意力参数
        self.a = nn.Parameter(torch.FloatTensor(n_heads, 2 * self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (n_nodes, n_nodes) 或 (batch, n_nodes, n_nodes)
            
        Returns:
            (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.size()
        
        # 线性变换
        h = self.W(x)  # (batch, n_nodes, head_dim * n_heads)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (batch, n_heads, n_nodes, head_dim)
        
        # 计算注意力分数
        # 拼接特征
        h_repeat = h.unsqueeze(3).repeat(1, 1, 1, n_nodes, 1)  # (batch, n_heads, n_nodes, n_nodes, head_dim)
        h_repeat_t = h.unsqueeze(2).repeat(1, 1, n_nodes, 1, 1)  # (batch, n_heads, n_nodes, n_nodes, head_dim)
        concat_h = torch.cat([h_repeat, h_repeat_t], dim=-1)  # (batch, n_heads, n_nodes, n_nodes, 2*head_dim)
        
        # 注意力分数
        e = torch.einsum('bnijd,hd->bnij', concat_h, self.a)  # (batch, n_heads, n_nodes, n_nodes)
        e = self.leaky_relu(e)
        
        # 使用邻接矩阵掩码
        if adj.dim() == 2:
            mask = adj.unsqueeze(0).unsqueeze(0)  # (1, 1, n_nodes, n_nodes)
        else:
            mask = adj.unsqueeze(1)  # (batch, 1, n_nodes, n_nodes)
        
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # softmax归一化
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        # 聚合
        h_prime = torch.matmul(attention, h)  # (batch, n_heads, n_nodes, head_dim)
        
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous()
            h_prime = h_prime.view(batch_size, n_nodes, -1)
        else:
            h_prime = h_prime.mean(dim=1)
        
        return h_prime


class AdaptiveAdjacency(nn.Module):
    """
    自适应邻接矩阵
    
    基于节点特征动态计算邻接矩阵
    """
    
    def __init__(self, n_nodes: int, hidden_dim: int):
        super().__init__()
        
        self.n_nodes = n_nodes
        
        # 学习节点嵌入
        self.node_embedding = nn.Parameter(torch.FloatTensor(n_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)
    
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        计算自适应邻接矩阵
        
        Args:
            x: 节点特征 (batch, n_nodes, features) - 可选
            
        Returns:
            邻接矩阵 (n_nodes, n_nodes) 或 (batch, n_nodes, n_nodes)
        """
        # 基于节点嵌入计算相似度
        adj = torch.mm(self.node_embedding, self.node_embedding.t())
        adj = F.softmax(adj, dim=1)
        
        # 如果提供了特征，可以结合特征相似度
        if x is not None:
            # 特征相似度
            x_norm = F.normalize(x, p=2, dim=-1)
            feat_adj = torch.bmm(x_norm, x_norm.transpose(1, 2))
            feat_adj = F.softmax(feat_adj, dim=-1)
            
            # 结合
            adj = adj.unsqueeze(0) * 0.5 + feat_adj * 0.5
        
        return adj


class GraphConvBranch(nn.Module):
    """
    图卷积分支
    
    将脑区作为节点，学习脑区间的信息传递
    """
    
    def __init__(self,
                 n_timesteps: int,
                 n_regions: int,
                 n_features: int,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 adj_type: Literal['predefined', 'learnable', 'adaptive'] = 'learnable',
                 use_gat: bool = False,
                 dropout: float = 0.3):
        """
        Args:
            n_timesteps: 时间步数 (T)
            n_regions: 脑区数/节点数 (R)
            n_features: 特征数 (F)
            hidden_dim: 隐藏层维度
            n_layers: 图卷积层数
            adj_type: 邻接矩阵类型
            use_gat: 是否使用图注意力
            dropout: Dropout率
        """
        super().__init__()
        
        self.n_timesteps = n_timesteps
        self.n_regions = n_regions
        self.n_features = n_features
        self.adj_type = adj_type
        
        # 邻接矩阵
        if adj_type == 'predefined':
            # 预定义邻接矩阵
            adj = torch.FloatTensor(PREDEFINED_ADJACENCY)
            self.register_buffer('adj', adj)
        elif adj_type == 'learnable':
            # 可学习邻接矩阵
            self.adj = nn.Parameter(torch.FloatTensor(n_regions, n_regions))
            nn.init.xavier_uniform_(self.adj)
        elif adj_type == 'adaptive':
            # 自适应邻接矩阵
            self.adj_module = AdaptiveAdjacency(n_regions, hidden_dim)
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # 图卷积层
        self.gcn_layers = nn.ModuleList()
        in_dim = hidden_dim
        
        for i in range(n_layers):
            out_dim = hidden_dim * (2 if i == n_layers - 1 else 1)
            if use_gat:
                self.gcn_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, n_heads=4, dropout=dropout)
                )
            else:
                self.gcn_layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        
        self.out_dim = in_dim
        
        # 时间聚合
        self.temporal_pool = nn.Sequential(
            nn.Linear(n_timesteps, 1),
            nn.Flatten()
        )
        
        # 节点聚合
        self.node_pool = nn.Sequential(
            nn.Linear(n_regions * self.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def get_adjacency(self, x: torch.Tensor = None) -> torch.Tensor:
        """获取邻接矩阵"""
        if self.adj_type == 'predefined':
            return self.adj
        elif self.adj_type == 'learnable':
            # 使用softmax确保非负和归一化
            return F.softmax(F.relu(self.adj), dim=1)
        elif self.adj_type == 'adaptive':
            return self.adj_module(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征张量 (batch, T, R, F)
            
        Returns:
            图特征向量 (batch, hidden_dim)
        """
        batch_size, T, R, F = x.size()
        
        # 输入投影
        x = self.input_proj(x)  # (batch, T, R, hidden_dim)
        
        # 获取邻接矩阵
        # 对于adaptive类型，使用平均特征计算邻接矩阵
        if self.adj_type == 'adaptive':
            x_mean = x.mean(dim=1)  # (batch, R, hidden_dim)
            adj = self.get_adjacency(x_mean)  # (batch, R, R)
        else:
            adj = self.get_adjacency()  # (R, R)
        
        # 对每个时间步进行图卷积
        outputs = []
        for t in range(T):
            h = x[:, t, :, :]  # (batch, R, hidden_dim)
            
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, adj)
                h = self.relu(h)
                h = self.dropout(h)
            
            outputs.append(h)  # (batch, R, out_dim)
        
        # 堆叠时间步: (batch, T, R, out_dim)
        x = torch.stack(outputs, dim=1)
        
        # 时间聚合: (batch, R, out_dim)
        x = x.permute(0, 2, 3, 1)  # (batch, R, out_dim, T)
        x = self.temporal_pool(x)  # (batch, R, out_dim)
        
        # 节点聚合: (batch, hidden_dim)
        x = x.view(batch_size, -1)  # (batch, R * out_dim)
        x = self.node_pool(x)  # (batch, hidden_dim)
        
        return x


class TripleBranchModel(nn.Module):
    """
    三分支模型：时间卷积 + 特征卷积 + 图卷积
    """
    
    def __init__(self,
                 n_timesteps: int = 8,
                 n_regions: int = 4,
                 n_features: int = 26,
                 n_classes: int = 3,
                 hidden_dim: int = 64,
                 fusion_type: Literal['concat', 'weighted', 'cross_attention'] = 'concat',
                 adj_type: Literal['predefined', 'learnable', 'adaptive'] = 'learnable',
                 use_graph_branch: bool = True,
                 use_gat: bool = False,
                 kernel_size: int = 3,
                 n_conv_layers: int = 2,
                 n_gcn_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            n_timesteps: 时间步数
            n_regions: 脑区数
            n_features: 特征数
            n_classes: 分类数
            hidden_dim: 隐藏层维度
            fusion_type: CNN分支融合方式
            adj_type: 邻接矩阵类型
            use_graph_branch: 是否使用图卷积分支
            use_gat: 是否使用图注意力
            kernel_size: CNN卷积核大小
            n_conv_layers: CNN卷积层数
            n_gcn_layers: 图卷积层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.use_graph_branch = use_graph_branch
        self.fusion_type = fusion_type
        
        # 导入双分支CNN的组件
        from dual_branch_cnn import (
            TemporalConvBranch, FeatureConvBranch,
            ConcatFusion, WeightedFusion, CrossAttentionFusion,
            ChannelAttention
        )
        
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
        
        # CNN融合层
        if fusion_type == 'concat':
            self.cnn_fusion = ConcatFusion(hidden_dim, hidden_dim, hidden_dim)
        elif fusion_type == 'weighted':
            self.cnn_fusion = WeightedFusion(hidden_dim, hidden_dim, hidden_dim)
        elif fusion_type == 'cross_attention':
            self.cnn_fusion = CrossAttentionFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # 图卷积分支（可选）
        if use_graph_branch:
            self.graph_branch = GraphConvBranch(
                n_timesteps=n_timesteps,
                n_regions=n_regions,
                n_features=n_features,
                hidden_dim=hidden_dim,
                n_layers=n_gcn_layers,
                adj_type=adj_type,
                use_gat=use_gat,
                dropout=dropout
            )
            
            # CNN和图特征融合
            self.final_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
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
        Args:
            x: 特征张量 (batch, T, R, F)
            
        Returns:
            分类logits (batch, n_classes)
        """
        # 时间和特征分支
        temporal_feat = self.temporal_branch(x)
        feature_feat = self.feature_branch(x)
        
        # CNN融合
        cnn_fused = self.cnn_fusion(temporal_feat, feature_feat)
        
        # 图卷积分支
        if self.use_graph_branch:
            graph_feat = self.graph_branch(x)
            # 最终融合
            fused = torch.cat([cnn_fused, graph_feat], dim=1)
            fused = self.final_fusion(fused)
        else:
            fused = cnn_fused
        
        # 注意力
        attended = self.attention(fused)
        
        # 分类
        logits = self.classifier(attended)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> dict:
        """
        获取各分支的特征（用于分析）
        """
        temporal_feat = self.temporal_branch(x)
        feature_feat = self.feature_branch(x)
        cnn_fused = self.cnn_fusion(temporal_feat, feature_feat)
        
        result = {
            'temporal': temporal_feat,
            'feature': feature_feat,
            'cnn_fused': cnn_fused
        }
        
        if self.use_graph_branch:
            graph_feat = self.graph_branch(x)
            result['graph'] = graph_feat
            fused = torch.cat([cnn_fused, graph_feat], dim=1)
            fused = self.final_fusion(fused)
            result['final_fused'] = fused
        
        return result


def create_model(n_timesteps: int = 8,
                 n_regions: int = 4,
                 n_features: int = 26,
                 n_classes: int = 3,
                 hidden_dim: int = 64,
                 fusion_type: str = 'concat',
                 use_graph: bool = True,
                 adj_type: str = 'learnable',
                 use_gat: bool = False,
                 **kwargs) -> TripleBranchModel:
    """
    创建模型的工厂函数
    """
    return TripleBranchModel(
        n_timesteps=n_timesteps,
        n_regions=n_regions,
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=hidden_dim,
        fusion_type=fusion_type,
        use_graph_branch=use_graph,
        adj_type=adj_type,
        use_gat=use_gat,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("图卷积分支测试")
    print("=" * 60)
    
    batch_size = 4
    n_timesteps = 8
    n_regions = 4
    n_features = 26
    
    x = torch.randn(batch_size, n_timesteps, n_regions, n_features)
    print(f"\n输入形状: {x.shape}")
    
    # 测试不同邻接矩阵类型
    for adj_type in ['predefined', 'learnable', 'adaptive']:
        print(f"\n{'='*40}")
        print(f"测试邻接矩阵类型: {adj_type}")
        print('='*40)
        
        branch = GraphConvBranch(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            hidden_dim=64,
            adj_type=adj_type
        )
        
        with torch.no_grad():
            out = branch(x)
        
        print(f"  输出形状: {out.shape}")
        
        # 显示邻接矩阵
        adj = branch.get_adjacency(x.mean(dim=1) if adj_type == 'adaptive' else None)
        if adj.dim() == 2:
            print(f"  邻接矩阵形状: {adj.shape}")
        else:
            print(f"  邻接矩阵形状: {adj.shape} (每个样本不同)")
    
    # 测试完整三分支模型
    print(f"\n{'='*60}")
    print("测试三分支模型")
    print('='*60)
    
    for use_graph in [False, True]:
        print(f"\n使用图分支: {use_graph}")
        
        model = create_model(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            n_classes=3,
            hidden_dim=64,
            fusion_type='concat',
            use_graph=use_graph,
            adj_type='learnable'
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")
        
        with torch.no_grad():
            logits = model(x)
        
        print(f"  输出形状: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

