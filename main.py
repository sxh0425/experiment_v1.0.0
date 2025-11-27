# -*- coding: utf-8 -*-
"""
主程序：EEG认知障碍分类特征提取

针对ds004504数据集进行：
1. S-Transform时频特征提取
2. CNN时频特征提取
3. 传统特征提取（RBP、频带比值、α峰值）
4. 特征拼接与保存

分类任务：AD vs FTD vs CN (三分类)

参数设置：
- 采样率：500Hz
- 窗口大小：4秒 (2000样本点)
- 重叠率：50%

脑区划分：
- 前额叶（Frontal）：Fp1, Fp2, F7, F3, Fz, F4, F8
- 颞叶（Temporal）：T3, T4, T5, T6
- 顶叶（Parietal）：P3, Pz, P4
- 枕叶（Occipital）：O1, O2

频带定义：
- δ (delta): 0.5-4 Hz
- θ (theta): 4-8 Hz
- α (alpha): 8-13 Hz
- β (beta): 13-30 Hz
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import EEGDataLoader, create_data_splits, LABEL_NAMES
from feature_extraction import (
    ComprehensiveFeatureExtractor, 
    extract_and_save_features,
    get_feature_summary
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='EEG认知障碍分类特征提取'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../dataset/ds004504',
        help='数据集路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./cache',
        help='输出目录'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=float,
        default=500.0,
        help='采样率 (Hz)'
    )
    
    parser.add_argument(
        '--window_size',
        type=float,
        default=4.0,
        help='窗口大小 (秒)'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='重叠率 (0-1)'
    )
    
    parser.add_argument(
        '--use_cnn',
        action='store_true',
        default=True,
        help='是否使用CNN提取特征'
    )
    
    parser.add_argument(
        '--no_cnn',
        action='store_true',
        help='不使用CNN提取特征'
    )
    
    parser.add_argument(
        '--cnn_feature_dim',
        type=int,
        default=128,
        help='CNN特征维度'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='计算设备'
    )
    
    parser.add_argument(
        '--n_freqs',
        type=int,
        default=64,
        help='S-Transform频率采样点数'
    )
    
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='测试模式（只处理少量数据）'
    )
    
    return parser.parse_args()


def print_config(args):
    """打印配置信息"""
    print("\n" + "=" * 70)
    print("配置参数")
    print("=" * 70)
    
    config = {
        '数据集路径': args.dataset_path,
        '输出目录': args.output_dir,
        '采样率': f"{args.sample_rate} Hz",
        '窗口大小': f"{args.window_size} 秒",
        '重叠率': f"{args.overlap * 100}%",
        '使用CNN': not args.no_cnn,
        'CNN特征维度': args.cnn_feature_dim,
        '计算设备': args.device,
        'S-Transform频率点数': args.n_freqs,
        '测试模式': args.test_mode
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("=" * 70 + "\n")
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 处理CNN参数
    use_cnn = args.use_cnn and not args.no_cnn
    
    # 打印配置
    config = print_config(args)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集路径
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path
    
    if not dataset_path.exists():
        print(f"错误：数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("步骤 1: 加载数据")
    print("=" * 70)
    
    # 创建数据加载器
    loader = EEGDataLoader(
        dataset_path=str(dataset_path),
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        overlap=args.overlap,
        use_preprocessed=True
    )
    
    # 显示数据集信息
    print(f"\n受试者数量: {len(loader.participants_df)}")
    print(f"  AD: {len(loader.get_subjects_by_group('AD'))}")
    print(f"  FTD: {len(loader.get_subjects_by_group('FTD'))}")
    print(f"  CN: {len(loader.get_subjects_by_group('CN'))}")
    
    # 加载数据
    if args.test_mode:
        print("\n[测试模式] 只加载部分数据...")
        # 只加载每组的前2个受试者
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
                
                print(f"  加载 {subject_id}: {n_windows} 个窗口")
            except Exception as e:
                print(f"  加载 {subject_id} 失败: {e}")
        
        all_segments = np.concatenate(all_segments, axis=0)
        all_labels = np.array(all_labels)
        all_subject_ids = np.array(all_subject_ids)
    else:
        all_segments, all_labels, channel_names, all_subject_ids = loader.load_all_subjects(
            verbose=True
        )
    
    print(f"\n数据形状: {all_segments.shape}")
    print(f"标签分布:")
    for label_id, label_name in LABEL_NAMES.items():
        count = np.sum(all_labels == label_id)
        print(f"  {label_name}: {count} 个窗口")
    
    print("\n" + "=" * 70)
    print("步骤 2: 特征提取")
    print("=" * 70)
    
    # 创建特征提取器
    extractor = ComprehensiveFeatureExtractor(
        sample_rate=args.sample_rate,
        n_freqs=args.n_freqs,
        use_cnn=use_cnn,
        cnn_feature_dim=args.cnn_feature_dim,
        device=args.device
    )
    
    # 提取特征
    print("\n开始提取特征...")
    features, feature_names = extractor.extract_batch_features(
        all_segments, 
        channel_names,
        verbose=True
    )
    
    print(f"\n特征矩阵形状: {features.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 特征统计
    summary = get_feature_summary(feature_names)
    print(f"\n特征类型统计:")
    print(f"  传统特征: {summary['traditional']}")
    print(f"  S-Transform特征: {summary['s_transform']}")
    print(f"  CNN特征: {summary['cnn']}")
    print(f"  总计: {summary['total']}")
    
    # 检查NaN
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    print(f"\nNaN数量: {nan_count}")
    print(f"Inf数量: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("警告：特征中存在NaN或Inf，进行替换处理...")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("\n" + "=" * 70)
    print("步骤 3: 保存结果")
    print("=" * 70)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存特征
    feature_file = output_dir / f'features_{timestamp}.pkl'
    result = {
        'features': features,
        'labels': all_labels,
        'feature_names': feature_names,
        'subject_ids': all_subject_ids,
        'channel_names': channel_names,
        'config': config,
        'feature_summary': summary
    }
    
    with open(feature_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\n特征已保存到: {feature_file}")
    
    # 保存配置信息
    config_file = output_dir / f'config_{timestamp}.json'
    with open(config_file, 'w', encoding='utf-8') as f:
        # 转换不可序列化的对象
        config_json = {k: str(v) for k, v in config.items()}
        config_json['feature_summary'] = summary
        config_json['n_samples'] = int(features.shape[0])
        config_json['n_features'] = int(features.shape[1])
        config_json['feature_names'] = feature_names
        json.dump(config_json, f, ensure_ascii=False, indent=2)
    
    print(f"配置已保存到: {config_file}")
    
    # 创建数据集划分
    print("\n" + "=" * 70)
    print("步骤 4: 创建数据集划分")
    print("=" * 70)
    
    splits = create_data_splits(
        all_subject_ids, 
        all_labels,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print(f"\n训练集: {len(splits['train'])} 个窗口 ({len(splits['train_subjects'])} 个受试者)")
    print(f"验证集: {len(splits['val'])} 个窗口 ({len(splits['val_subjects'])} 个受试者)")
    print(f"测试集: {len(splits['test'])} 个窗口 ({len(splits['test_subjects'])} 个受试者)")
    
    # 保存划分信息
    splits_file = output_dir / f'splits_{timestamp}.pkl'
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"\n数据集划分已保存到: {splits_file}")
    
    print("\n" + "=" * 70)
    print("特征提取完成！")
    print("=" * 70)
    
    # 显示特征示例
    print("\n特征示例（前10个特征）:")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {name}: mean={np.mean(features[:, i]):.4f}, std={np.std(features[:, i]):.4f}")
    
    return features, all_labels, feature_names


if __name__ == "__main__":
    main()


