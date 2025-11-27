# -*- coding: utf-8 -*-
"""
训练脚本

支持双分支CNN和三分支（含图卷积）模型的训练与评估
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 添加当前目录
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import EEGDataLoader, LABEL_NAMES
from subwindow_features import SubwindowFeatureExtractor, REGION_ORDER
from dual_branch_cnn import create_model as create_dual_branch_model, DualBranchCNN
from graph_conv import create_model as create_triple_branch_model, TripleBranchModel
from dataset import create_dataloaders, EEGFeatureDataset


class Trainer:
    """
    模型训练器
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 class_weights: Optional[torch.Tensor] = None):
        """
        Args:
            model: 模型
            device: 计算设备
            class_weights: 类别权重
        """
        self.model = model.to(device)
        self.device = device
        
        # 损失函数
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, 
                    dataloader, 
                    optimizer) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def fit(self,
            train_loader,
            val_loader,
            n_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            patience: int = 15,
            min_delta: float = 1e-4,
            save_dir: Optional[str] = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            min_delta: 最小改善值
            save_dir: 模型保存目录
        """
        # 优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 早停
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        # 训练循环
        print("\n开始训练...")
        print("=" * 70)
        
        for epoch in range(n_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # 更新学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # 打印进度
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # 检查是否改善
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                if save_dir:
                    self.save_checkpoint(save_dir, epoch, val_loss, val_acc)
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发! 在 epoch {epoch+1}")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print("=" * 70)
        print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int, 
                        val_loss: float, val_acc: float):
        """保存检查点"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        torch.save(checkpoint, save_dir / 'best_model.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def evaluate_model(model: nn.Module,
                   test_loader,
                   device: str = 'cuda') -> Dict:
    """
    全面评估模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_per_class': precision_score(all_labels, all_preds, average=None).tolist(),
        'recall_per_class': recall_score(all_labels, all_preds, average=None).tolist(),
        'f1_per_class': f1_score(all_labels, all_preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    # 打印报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=['AD', 'FTD', 'CN']
    ))
    
    return metrics, all_preds, all_labels, all_probs


def plot_training_history(history: Dict, save_path: str):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    axes[2].plot(history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['AD', 'FTD', 'CN']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 添加数字标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双分支CNN训练')
    
    parser.add_argument('--dataset_path', type=str, 
                        default='../dataset/ds004504',
                        help='数据集路径')
    
    parser.add_argument('--output_dir', type=str,
                        default='./results',
                        help='输出目录')
    
    parser.add_argument('--sample_rate', type=float, default=500.0)
    parser.add_argument('--window_size', type=float, default=4.0)
    parser.add_argument('--subwindow_size', type=float, default=0.5)
    parser.add_argument('--subwindow_overlap', type=float, default=0.5)
    
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'weighted', 'cross_attention'],
                        help='CNN分支融合方式')
    
    # 图卷积相关参数
    parser.add_argument('--use_graph', action='store_true',
                        help='使用图卷积分支')
    
    parser.add_argument('--adj_type', type=str, default='learnable',
                        choices=['predefined', 'learnable', 'adaptive'],
                        help='邻接矩阵类型: predefined(预定义), learnable(可学习), adaptive(自适应)')
    
    parser.add_argument('--use_gat', action='store_true',
                        help='使用图注意力网络(GAT)替代GCN')
    
    parser.add_argument('--n_gcn_layers', type=int, default=2,
                        help='图卷积层数')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_conv_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'])
    
    parser.add_argument('--test_mode', action='store_true',
                        help='测试模式')
    
    parser.add_argument('--use_cached_features', action='store_true',
                        help='使用缓存的特征')
    
    parser.add_argument('--cached_features_path', type=str,
                        default='./cache/subwindow_features.pkl',
                        help='缓存特征路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = 'cpu'
    
    model_type = "三分支CNN+GCN" if args.use_graph else "双分支CNN"
    
    print("=" * 70)
    print(f"{model_type} 训练")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  模型类型: {model_type}")
    print(f"  CNN融合方式: {args.fusion_type}")
    if args.use_graph:
        print(f"  邻接矩阵类型: {args.adj_type}")
        print(f"  使用GAT: {args.use_gat}")
        print(f"  GCN层数: {args.n_gcn_layers}")
    print(f"  隐藏维度: {args.hidden_dim}")
    print(f"  CNN卷积层数: {args.n_conv_layers}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  设备: {device}")
    print(f"  输出目录: {output_dir}")
    
    # 加载或提取特征
    if args.use_cached_features and Path(args.cached_features_path).exists():
        print(f"\n加载缓存特征: {args.cached_features_path}")
        with open(args.cached_features_path, 'rb') as f:
            cache = pickle.load(f)
        
        features = cache['features']
        labels = cache['labels']
        subject_ids = cache['subject_ids']
        feature_info = cache.get('info', {})
        channel_names = cache.get('channel_names', None)
    else:
        print("\n步骤1: 加载EEG数据")
        print("-" * 40)
        
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).parent / dataset_path
        
        loader = EEGDataLoader(
            dataset_path=str(dataset_path),
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            overlap=0.5,
            use_preprocessed=True
        )
        
        if args.test_mode:
            print("[测试模式] 只加载部分数据")
            test_subjects = (
                loader.get_subjects_by_group('AD')[:2] +
                loader.get_subjects_by_group('FTD')[:2] +
                loader.get_subjects_by_group('CN')[:2]
            )
            
            all_segments = []
            all_labels = []
            all_subject_ids = []
            channel_names = None
            
            for subject_id in test_subjects:
                try:
                    data, ch_names, label = loader.load_single_subject(subject_id)
                    if channel_names is None:
                        channel_names = ch_names
                    segments = loader.segment_signal(data)
                    n_windows = segments.shape[0]
                    all_segments.append(segments)
                    all_labels.extend([label] * n_windows)
                    all_subject_ids.extend([subject_id] * n_windows)
                except Exception as e:
                    print(f"  加载 {subject_id} 失败: {e}")
            
            all_segments = np.concatenate(all_segments, axis=0)
            all_labels = np.array(all_labels)
            all_subject_ids = np.array(all_subject_ids)
        else:
            all_segments, all_labels, channel_names, all_subject_ids = loader.load_all_subjects()
        
        print(f"\nEEG数据形状: {all_segments.shape}")
        
        # 提取子窗口特征
        print("\n步骤2: 提取子窗口特征")
        print("-" * 40)
        
        extractor = SubwindowFeatureExtractor(
            sample_rate=args.sample_rate,
            window_size=args.window_size,
            subwindow_size=args.subwindow_size,
            subwindow_overlap=args.subwindow_overlap
        )
        
        feature_info = extractor.get_info()
        print(f"  时间步数(T): {feature_info['n_timesteps']}")
        print(f"  脑区数(R): {feature_info['n_regions']}")
        print(f"  特征数(F): {feature_info['n_features']}")
        
        features = extractor.extract_batch_features(
            all_segments, 
            channel_names,
            verbose=True
        )
        
        labels = all_labels
        subject_ids = all_subject_ids
        
        # 缓存特征
        cache_dir = Path(args.cached_features_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        with open(args.cached_features_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'labels': labels,
                'subject_ids': subject_ids,
                'channel_names': channel_names,
                'info': feature_info
            }, f)
        
        print(f"\n特征已缓存到: {args.cached_features_path}")
    
    print(f"\n特征张量形状: {features.shape}")
    
    # 处理NaN
    if np.any(np.isnan(features)):
        print("警告: 特征中存在NaN，进行替换")
        features = np.nan_to_num(features, nan=0.0)
    
    # 创建数据加载器
    print("\n步骤3: 创建数据加载器")
    print("-" * 40)
    
    dataloaders = create_dataloaders(
        features=features,
        labels=labels,
        subject_ids=subject_ids,
        batch_size=args.batch_size
    )
    
    # 创建模型
    print("\n步骤4: 创建模型")
    print("-" * 40)
    
    n_timesteps = features.shape[1]
    n_regions = features.shape[2]
    n_features = features.shape[3]
    
    if args.use_graph:
        # 三分支模型（时间卷积 + 特征卷积 + 图卷积）
        print(f"  创建三分支模型（含图卷积）")
        model = create_triple_branch_model(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            n_classes=3,
            hidden_dim=args.hidden_dim,
            fusion_type=args.fusion_type,
            use_graph=True,
            adj_type=args.adj_type,
            use_gat=args.use_gat,
            n_conv_layers=args.n_conv_layers,
            n_gcn_layers=args.n_gcn_layers,
            dropout=args.dropout
        )
    else:
        # 双分支模型（时间卷积 + 特征卷积）
        print(f"  创建双分支模型")
        model = create_dual_branch_model(
            n_timesteps=n_timesteps,
            n_regions=n_regions,
            n_features=n_features,
            n_classes=3,
            hidden_dim=args.hidden_dim,
            fusion_type=args.fusion_type,
            n_conv_layers=args.n_conv_layers,
            dropout=args.dropout
        )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 训练
    print("\n步骤5: 训练模型")
    print("-" * 40)
    
    trainer = Trainer(
        model=model,
        device=device,
        class_weights=dataloaders['class_weights']
    )
    
    history = trainer.fit(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=str(output_dir / 'checkpoints')
    )
    
    # 评估
    print("\n步骤6: 测试集评估")
    print("-" * 40)
    
    metrics, preds, labels_test, probs = evaluate_model(
        model=model,
        test_loader=dataloaders['test'],
        device=device
    )
    
    # 保存结果
    print("\n步骤7: 保存结果")
    print("-" * 40)
    
    # 保存训练历史图
    plot_training_history(history, str(output_dir / 'training_history.png'))
    print(f"  训练历史图: {output_dir / 'training_history.png'}")
    
    # 保存混淆矩阵
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, str(output_dir / 'confusion_matrix.png'))
    print(f"  混淆矩阵图: {output_dir / 'confusion_matrix.png'}")
    
    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  指标文件: {output_dir / 'metrics.json'}")
    
    # 保存配置
    config = vars(args)
    config['n_timesteps'] = n_timesteps
    config['n_regions'] = n_regions
    config['n_features'] = n_features
    config['total_params'] = total_params
    config['model_type'] = "triple_branch" if args.use_graph else "dual_branch"
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  配置文件: {output_dir / 'config.json'}")
    
    # 保存历史
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n最终结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数(macro): {metrics['f1_macro']:.4f}")
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()

