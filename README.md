# EEG认知障碍分类

基于多分支卷积神经网络的EEG认知障碍三分类系统（AD vs FTD vs CN）

## 项目概述

本项目实现了一个创新的EEG分类框架，支持**双分支**和**三分支**两种模式：

1. **子窗口特征提取** - 保留时间维度，生成3D特征张量 (T × R × F)
2. **时间卷积分支** - 沿时间维度卷积，捕捉时序动态
3. **特征卷积分支** - 沿特征维度卷积，捕捉特征关系
4. **图卷积分支（可选）** - 将脑区作为节点，学习脑区间连接性
5. **多种融合方式** - 支持拼接、加权、交叉注意力三种融合
6. **注意力机制** - 加权重要特征

## 架构设计

### 双分支模式（默认）
```
                    EEG信号 (4秒, 19通道)
                              │
                    ┌─────────┴─────────┐
                    │   子窗口划分       │
                    │  (8个×0.5秒)       │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   特征提取         │
                    │  (传统+S-Transform)│
                    └─────────┬─────────┘
                              │
                    特征张量 (T=8, R=4, F=26)
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌─────────────────┐             ┌─────────────────┐
    │  时间卷积分支    │             │  特征卷积分支    │
    │  Conv1D(时间)   │             │  Conv1D(特征)   │
    │  捕捉时序动态   │             │  捕捉特征关系   │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             └───────────┬───────────────────┘
                         │
                ┌────────┴────────┐
                │    特征融合      │
                └────────┬────────┘
                         │
                   注意力 + FC
                         │
                   3分类输出
```

### 三分支模式（--use_graph）
```
                    特征张量 (T=8, R=4, F=26)
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  时间卷积分支  │     │  特征卷积分支  │     │  图卷积分支    │
│  Conv1D(时间) │     │  Conv1D(特征) │     │  GCN/GAT      │
│  捕捉时序动态 │     │  捕捉特征关系 │     │  捕捉脑区关系 │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └──────────┬──────────┘                     │
                   │                                │
            ┌──────┴──────┐                         │
            │  CNN融合     │                         │
            └──────┬──────┘                         │
                   │                                │
                   └────────────┬───────────────────┘
                                │
                         ┌──────┴──────┐
                         │  最终融合    │
                         └──────┬──────┘
                                │
                          注意力 + FC
                                │
                          3分类输出
```

## 特征设计

### 子窗口划分
- 4秒窗口 → 8个子窗口（每个0.5秒，50%重叠）
- 保留时间维度信息

### 每个子窗口提取的特征（共26个）

| 类别 | 特征 | 数量 |
|------|------|------|
| RBP | δ, θ, α, β 相对功率 | 4 |
| 比值 | θ/α, θ/β, (δ+θ)/(α+β) | 3 |
| α峰值 | 峰值频率、功率、带宽、中心频率 | 4 |
| S-Transform | 全局统计 + 频带统计 + 频率质心 + 时间熵 | 15 |

### 脑区划分

| 脑区 | 电极 | 数量 |
|------|------|------|
| 前额叶 (Frontal) | Fp1, Fp2, F7, F3, Fz, F4, F8 | 7 |
| 颞叶 (Temporal) | T3, T4, T5, T6 | 4 |
| 顶叶 (Parietal) | P3, Pz, P4 | 3 |
| 枕叶 (Occipital) | O1, O2 | 2 |

### 输出特征张量
```
形状: (batch, T=8, R=4, F=26)
  - T: 8个时间步
  - R: 4个脑区
  - F: 26个特征
```

## 融合方式

### CNN分支融合 (--fusion_type)

| 方式 | 参数值 | 描述 |
|------|--------|------|
| A. 简单拼接 | `concat` | 直接拼接两个分支的特征 |
| B. 加权融合 | `weighted` | 学习权重平衡两个分支 |
| C. 交叉注意力 | `cross_attention` | 双向注意力融合 |

## 图卷积分支

### 邻接矩阵类型 (--adj_type)

| 类型 | 参数值 | 描述 |
|------|--------|------|
| 预定义 | `predefined` | 基于解剖学连接的固定邻接矩阵 |
| 可学习 | `learnable` | 网络自动学习邻接矩阵权重 |
| 自适应 | `adaptive` | 基于特征相似度动态计算邻接矩阵 |

### 图神经网络类型 (--use_gat)

| 类型 | 描述 |
|------|------|
| GCN（默认） | 标准图卷积网络 |
| GAT | 图注意力网络，动态学习节点间注意力权重 |

### 预定义邻接矩阵

基于脑区解剖学连接：

```
            Frontal  Temporal  Parietal  Occipital
Frontal      1.0       0.7       0.5       0.3
Temporal     0.7       1.0       0.6       0.4
Parietal     0.5       0.6       1.0       0.8
Occipital    0.3       0.4       0.8       1.0
```

## 安装

```bash
# 创建环境
conda create -n eeg_cnn python=3.9
conda activate eeg_cnn

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
# ============ 双分支模式（默认）============
# 使用拼接融合
python train.py --dataset_path ../dataset/ds004504 --fusion_type concat

# 使用加权融合
python train.py --dataset_path ../dataset/ds004504 --fusion_type weighted

# 使用交叉注意力融合
python train.py --dataset_path ../dataset/ds004504 --fusion_type cross_attention

# ============ 三分支模式（含图卷积）============
# 使用图卷积 + 可学习邻接矩阵
python train.py --dataset_path ../dataset/ds004504 --use_graph

# 使用预定义邻接矩阵
python train.py --dataset_path ../dataset/ds004504 --use_graph --adj_type predefined

# 使用自适应邻接矩阵
python train.py --dataset_path ../dataset/ds004504 --use_graph --adj_type adaptive

# 使用图注意力网络(GAT)
python train.py --dataset_path ../dataset/ds004504 --use_graph --use_gat

# 完整配置示例
python train.py --dataset_path ../dataset/ds004504 \
    --use_graph --adj_type learnable --use_gat \
    --fusion_type cross_attention --hidden_dim 64

# ============ 其他选项 ============
# 测试模式（快速验证）
python train.py --dataset_path ../dataset/ds004504 --test_mode

# 使用缓存特征（加速）
python train.py --use_cached_features --cached_features_path ./cache/subwindow_features.pkl
```

### 2. 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset_path` | `../dataset/ds004504` | 数据集路径 |
| `--output_dir` | `./results` | 输出目录 |
| `--fusion_type` | `concat` | CNN融合方式 |
| `--use_graph` | `False` | 启用图卷积分支 |
| `--adj_type` | `learnable` | 邻接矩阵类型 |
| `--use_gat` | `False` | 使用图注意力网络 |
| `--n_gcn_layers` | `2` | 图卷积层数 |
| `--hidden_dim` | `64` | 隐藏层维度 |
| `--n_conv_layers` | `2` | CNN卷积层数 |
| `--batch_size` | `32` | 批大小 |
| `--n_epochs` | `100` | 训练轮数 |
| `--learning_rate` | `1e-3` | 学习率 |
| `--patience` | `15` | 早停耐心值 |
| `--device` | `cuda` | 计算设备 |

### 3. Python API

```python
from subwindow_features import SubwindowFeatureExtractor
from dual_branch_cnn import create_model

# 创建特征提取器
extractor = SubwindowFeatureExtractor(
    sample_rate=500.0,
    window_size=4.0,
    subwindow_size=0.5,
    subwindow_overlap=0.5
)

# 提取特征
features = extractor.extract_all_features(eeg_data, channel_names)
print(f"特征形状: {features.shape}")  # (8, 4, 26)

# 创建模型
model = create_model(
    n_timesteps=8,
    n_regions=4,
    n_features=26,
    n_classes=3,
    fusion_type='cross_attention'
)

# 前向传播
logits = model(torch.FloatTensor(features).unsqueeze(0))
```

## 项目结构

```
experiment_v1.0.0/
├── train.py                  # 训练脚本（主入口）
├── data_loader.py            # 数据加载
├── dataset.py                # PyTorch数据集类
├── subwindow_features.py     # 子窗口特征提取（保留时间维度）
├── dual_branch_cnn.py        # 双分支CNN模型（时间+特征卷积）
├── graph_conv.py             # 图卷积模块（GCN/GAT）
├── s_transform.py            # S-Transform实现
├── traditional_features.py   # 传统特征提取
├── main.py                   # 原始特征提取（1D向量）
├── cnn_feature_extractor.py  # 原始CNN特征提取器
├── feature_extraction.py     # 综合特征提取
├── requirements.txt          # 依赖包
├── README.md                 # 说明文档
├── cache/                    # 特征缓存
└── results/                  # 训练结果
```

## 输出文件

训练完成后在 `results/run_YYYYMMDD_HHMMSS/` 目录生成：

- `checkpoints/best_model.pth` - 最佳模型权重
- `training_history.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵图
- `metrics.json` - 评估指标
- `config.json` - 训练配置
- `history.json` - 训练历史

## 模型对比

| 模型 | 参数 | 描述 |
|------|------|------|
| 双分支CNN | `--fusion_type concat` | 时间卷积 + 特征卷积，简单拼接 |
| 双分支CNN（加权） | `--fusion_type weighted` | 学习分支权重 |
| 双分支CNN（注意力） | `--fusion_type cross_attention` | 交叉注意力融合 |
| 三分支CNN+GCN | `--use_graph` | 添加图卷积分支 |
| 三分支CNN+GAT | `--use_graph --use_gat` | 使用图注意力网络 |

## 标签说明

| 标签值 | 组别 | 描述 |
|--------|------|------|
| 0 | AD | 阿尔茨海默病 |
| 1 | FTD | 额颞叶痴呆 |
| 2 | CN | 认知正常 |

## License

MIT License
