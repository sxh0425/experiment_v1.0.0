# -*- coding: utf-8 -*-
"""
数据加载模块
用于加载和预处理 ds004504 数据集的 EEG 数据

数据集信息：
- 88个受试者：36个AD，23个FTD，29个CN
- 采样率：500Hz
- 19个电极（10-20系统）
- 数据格式：.set (EEGLAB格式)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mne
import warnings

warnings.filterwarnings('ignore')


# 电极配置
CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# 脑区划分
BRAIN_REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],  # 前额叶
    'temporal': ['T3', 'T4', 'T5', 'T6'],  # 颞叶
    'parietal': ['P3', 'Pz', 'P4'],  # 顶叶
    'occipital': ['O1', 'O2']  # 枕叶
}

# 中央区（可选，用于参考）
CENTRAL_CHANNELS = ['C3', 'Cz', 'C4']

# 标签映射
LABEL_MAP = {
    'A': 0,  # AD (Alzheimer's Disease)
    'F': 1,  # FTD (Frontotemporal Dementia)
    'C': 2   # CN (Cognitively Normal / Healthy Control)
}

LABEL_NAMES = {
    0: 'AD',
    1: 'FTD',
    2: 'CN'
}


class EEGDataLoader:
    """
    EEG数据加载器
    
    用于加载ds004504数据集，支持窗口分割和标签提取
    """
    
    def __init__(self, 
                 dataset_path: str,
                 sample_rate: float = 500.0,
                 window_size: float = 4.0,
                 overlap: float = 0.5,
                 use_preprocessed: bool = True):
        """
        初始化数据加载器
        
        Args:
            dataset_path: 数据集路径 (ds004504文件夹)
            sample_rate: 采样率 (Hz)
            window_size: 窗口大小 (秒)
            overlap: 重叠率 (0-1)
            use_preprocessed: 是否使用预处理后的数据
        """
        self.dataset_path = Path(dataset_path)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.use_preprocessed = use_preprocessed
        
        # 计算窗口参数
        self.window_samples = int(window_size * sample_rate)  # 4s * 500Hz = 2000
        self.step_samples = int(self.window_samples * (1 - overlap))  # 50% overlap = 1000
        
        # 加载参与者信息
        self.participants_df = self._load_participants_info()
        
        # 频带定义
        self.bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0)
        }
    
    def _load_participants_info(self) -> pd.DataFrame:
        """
        加载参与者信息
        
        Returns:
            包含参与者信息的DataFrame
        """
        participants_file = self.dataset_path / 'participants.tsv'
        
        if not participants_file.exists():
            raise FileNotFoundError(f"找不到参与者信息文件: {participants_file}")
        
        df = pd.read_csv(participants_file, sep='\t')
        
        # 添加数值标签
        df['label'] = df['Group'].map(LABEL_MAP)
        df['label_name'] = df['label'].map(LABEL_NAMES)
        
        return df
    
    def _get_eeg_file_path(self, subject_id: str) -> Path:
        """
        获取EEG文件路径
        
        Args:
            subject_id: 受试者ID (如 'sub-001')
            
        Returns:
            EEG文件路径
        """
        if self.use_preprocessed:
            # 使用预处理后的数据
            eeg_path = self.dataset_path / 'preprocess' / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
        else:
            # 使用原始数据
            eeg_path = self.dataset_path / subject_id / 'eeg' / f'{subject_id}_task-eyesclosed_eeg.set'
        
        return eeg_path
    
    def load_single_subject(self, subject_id: str) -> Tuple[np.ndarray, List[str], int]:
        """
        加载单个受试者的EEG数据
        
        Args:
            subject_id: 受试者ID
            
        Returns:
            data: EEG数据 (channels x samples)
            channel_names: 通道名称列表
            label: 标签
        """
        eeg_path = self._get_eeg_file_path(subject_id)
        
        if not eeg_path.exists():
            raise FileNotFoundError(f"找不到EEG文件: {eeg_path}")
        
        # 使用MNE读取.set文件
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        
        # 获取数据和通道名称
        data = raw.get_data()
        channel_names = raw.ch_names
        
        # 获取标签
        subject_info = self.participants_df[self.participants_df['participant_id'] == subject_id]
        if len(subject_info) == 0:
            raise ValueError(f"找不到受试者信息: {subject_id}")
        
        label = subject_info['label'].values[0]
        
        return data, channel_names, label
    
    def segment_signal(self, data: np.ndarray) -> np.ndarray:
        """
        将连续信号分割成固定大小的窗口
        
        Args:
            data: EEG数据 (channels x samples)
            
        Returns:
            segments: 分割后的数据 (n_windows x channels x window_samples)
        """
        n_channels, n_samples = data.shape
        
        # 计算窗口数量
        n_windows = (n_samples - self.window_samples) // self.step_samples + 1
        
        if n_windows <= 0:
            raise ValueError(f"信号长度不足以生成窗口: {n_samples} samples")
        
        # 分割信号
        segments = np.zeros((n_windows, n_channels, self.window_samples))
        
        for i in range(n_windows):
            start = i * self.step_samples
            end = start + self.window_samples
            segments[i] = data[:, start:end]
        
        return segments
    
    def get_channel_indices(self, channel_names: List[str], target_channels: List[str]) -> List[int]:
        """
        获取目标通道的索引
        
        Args:
            channel_names: 所有通道名称
            target_channels: 目标通道名称
            
        Returns:
            通道索引列表
        """
        indices = []
        for ch in target_channels:
            if ch in channel_names:
                indices.append(channel_names.index(ch))
        return indices
    
    def get_region_data(self, data: np.ndarray, channel_names: List[str], 
                        region: str) -> np.ndarray:
        """
        获取特定脑区的数据
        
        Args:
            data: EEG数据 (... x channels x samples)
            channel_names: 通道名称
            region: 脑区名称 ('frontal', 'temporal', 'parietal', 'occipital')
            
        Returns:
            该脑区的数据
        """
        if region not in BRAIN_REGIONS:
            raise ValueError(f"未知脑区: {region}")
        
        region_channels = BRAIN_REGIONS[region]
        indices = self.get_channel_indices(channel_names, region_channels)
        
        if len(data.shape) == 2:
            return data[indices, :]
        elif len(data.shape) == 3:
            return data[:, indices, :]
        else:
            raise ValueError(f"不支持的数据维度: {data.shape}")
    
    def load_all_subjects(self, 
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        加载所有受试者的数据
        
        Args:
            verbose: 是否显示进度
            
        Returns:
            all_segments: 所有窗口数据 (n_total_windows x channels x window_samples)
            all_labels: 所有窗口的标签
            channel_names: 通道名称
            subject_ids: 每个窗口对应的受试者ID
        """
        all_segments = []
        all_labels = []
        all_subject_ids = []
        channel_names = None
        
        n_subjects = len(self.participants_df)
        
        for idx, row in self.participants_df.iterrows():
            subject_id = row['participant_id']
            
            if verbose:
                print(f"加载 {subject_id} ({idx+1}/{n_subjects})...")
            
            try:
                data, ch_names, label = self.load_single_subject(subject_id)
                
                if channel_names is None:
                    channel_names = ch_names
                
                # 分割信号
                segments = self.segment_signal(data)
                n_windows = segments.shape[0]
                
                all_segments.append(segments)
                all_labels.extend([label] * n_windows)
                all_subject_ids.extend([subject_id] * n_windows)
                
                if verbose:
                    print(f"  -> {n_windows} 个窗口, 标签: {LABEL_NAMES[label]}")
                    
            except Exception as e:
                print(f"  -> 加载失败: {e}")
                continue
        
        # 合并所有数据
        all_segments = np.concatenate(all_segments, axis=0)
        all_labels = np.array(all_labels)
        all_subject_ids = np.array(all_subject_ids)
        
        if verbose:
            print(f"\n总共加载 {all_segments.shape[0]} 个窗口")
            print(f"数据形状: {all_segments.shape}")
            print(f"标签分布: AD={np.sum(all_labels==0)}, FTD={np.sum(all_labels==1)}, CN={np.sum(all_labels==2)}")
        
        return all_segments, all_labels, channel_names, all_subject_ids
    
    def get_subject_info(self, subject_id: str) -> Dict:
        """
        获取受试者的详细信息
        
        Args:
            subject_id: 受试者ID
            
        Returns:
            受试者信息字典
        """
        info = self.participants_df[self.participants_df['participant_id'] == subject_id]
        
        if len(info) == 0:
            raise ValueError(f"找不到受试者: {subject_id}")
        
        return info.iloc[0].to_dict()
    
    def get_subjects_by_group(self, group: str) -> List[str]:
        """
        获取特定组别的所有受试者ID
        
        Args:
            group: 组别 ('A', 'F', 'C' 或 'AD', 'FTD', 'CN')
            
        Returns:
            受试者ID列表
        """
        # 支持不同格式的组别名称
        group_map = {'AD': 'A', 'FTD': 'F', 'CN': 'C'}
        if group in group_map:
            group = group_map[group]
        
        return self.participants_df[self.participants_df['Group'] == group]['participant_id'].tolist()


def create_data_splits(subject_ids: np.ndarray, 
                       labels: np.ndarray,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Dict:
    """
    创建数据集划分（按受试者划分，避免数据泄露）
    
    Args:
        subject_ids: 每个样本对应的受试者ID
        labels: 样本标签
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        包含训练/验证/测试集索引的字典
    """
    from sklearn.model_selection import train_test_split
    
    np.random.seed(random_state)
    
    # 获取唯一的受试者及其标签
    unique_subjects = np.unique(subject_ids)
    subject_labels = []
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        subject_labels.append(labels[mask][0])
    
    subject_labels = np.array(subject_labels)
    
    # 按受试者划分
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, 
        test_size=test_size, 
        stratify=subject_labels,
        random_state=random_state
    )
    
    # 获取训练+验证集的标签
    train_val_labels = []
    for subj in train_val_subjects:
        mask = subject_ids == subj
        train_val_labels.append(labels[mask][0])
    train_val_labels = np.array(train_val_labels)
    
    # 从训练+验证集中划分验证集
    val_ratio = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    # 获取样本索引
    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects
    }


if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 设置数据集路径
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'ds004504'
    
    if not dataset_path.exists():
        print(f"数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 创建数据加载器
    loader = EEGDataLoader(
        dataset_path=str(dataset_path),
        sample_rate=500.0,
        window_size=4.0,
        overlap=0.5,
        use_preprocessed=True
    )
    
    print("=" * 50)
    print("数据集信息:")
    print("=" * 50)
    print(f"受试者数量: {len(loader.participants_df)}")
    print(f"AD: {len(loader.get_subjects_by_group('AD'))}")
    print(f"FTD: {len(loader.get_subjects_by_group('FTD'))}")
    print(f"CN: {len(loader.get_subjects_by_group('CN'))}")
    
    print("\n" + "=" * 50)
    print("窗口参数:")
    print("=" * 50)
    print(f"窗口大小: {loader.window_size}s ({loader.window_samples} samples)")
    print(f"重叠率: {loader.overlap * 100}%")
    print(f"步长: {loader.step_samples} samples")
    
    print("\n" + "=" * 50)
    print("脑区划分:")
    print("=" * 50)
    for region, channels in BRAIN_REGIONS.items():
        print(f"{region}: {channels}")
    
    # 测试加载单个受试者
    print("\n" + "=" * 50)
    print("测试加载单个受试者:")
    print("=" * 50)
    
    try:
        subject_id = 'sub-001'
        data, ch_names, label = loader.load_single_subject(subject_id)
        print(f"受试者: {subject_id}")
        print(f"数据形状: {data.shape}")
        print(f"通道数: {len(ch_names)}")
        print(f"通道名称: {ch_names}")
        print(f"标签: {label} ({LABEL_NAMES[label]})")
        
        # 测试分割
        segments = loader.segment_signal(data)
        print(f"分割后形状: {segments.shape}")
        
        # 测试获取脑区数据
        for region in BRAIN_REGIONS.keys():
            region_data = loader.get_region_data(segments, ch_names, region)
            print(f"{region} 区域数据形状: {region_data.shape}")
            
    except Exception as e:
        print(f"加载失败: {e}")


