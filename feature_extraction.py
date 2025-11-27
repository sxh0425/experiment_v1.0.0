# -*- coding: utf-8 -*-
"""
综合特征提取模块

整合以下特征：
1. S-Transform 时频特征
2. CNN 提取的时频特征
3. 传统特征（RBP、频带比值、α峰值）

按四个脑区分别提取并拼接
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import warnings
from tqdm import tqdm

from s_transform import StockwellTransform, stockwell_transform_batch
from traditional_features import (
    TraditionalFeatureExtractor, 
    InterRegionFeatures,
    extract_traditional_features,
    BRAIN_REGIONS,
    FREQUENCY_BANDS
)
from cnn_feature_extractor import (
    StockwellCNNFeatureExtractor,
    prepare_stockwell_input,
    extract_cnn_features
)

warnings.filterwarnings('ignore')


class ComprehensiveFeatureExtractor:
    """
    综合特征提取器
    
    整合所有类型的特征：
    - S-Transform 时频统计特征
    - CNN 深度特征
    - 传统频域特征
    """
    
    def __init__(self,
                 sample_rate: float = 500.0,
                 freq_range: Tuple[float, float] = (0.5, 45.0),
                 n_freqs: int = 64,
                 use_cnn: bool = True,
                 cnn_feature_dim: int = 128,
                 device: str = 'cpu'):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率
            freq_range: 频率范围
            n_freqs: S-Transform频率采样点数
            use_cnn: 是否使用CNN提取特征
            cnn_feature_dim: CNN特征维度
            device: 计算设备
        """
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.n_freqs = n_freqs
        self.use_cnn = use_cnn
        self.cnn_feature_dim = cnn_feature_dim
        self.device = device
        
        # 初始化各个提取器
        self.st = StockwellTransform(sample_rate)
        self.traditional_extractor = TraditionalFeatureExtractor(sample_rate)
        self.inter_region_extractor = InterRegionFeatures(sample_rate)
        
        # CNN模型（延迟初始化）
        self.cnn_model = None
        
        # 脑区定义
        self.regions = BRAIN_REGIONS
        self.bands = FREQUENCY_BANDS
    
    def _init_cnn_model(self, n_channels: int, n_times: int):
        """
        初始化CNN模型
        """
        self.cnn_model = StockwellCNNFeatureExtractor(
            n_freqs=self.n_freqs,
            n_times=n_times,
            n_channels=n_channels,
            feature_dim=self.cnn_feature_dim
        )
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def extract_st_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        从单通道信号提取S-Transform统计特征
        
        Args:
            signal: 1D信号
            
        Returns:
            S-Transform特征字典
        """
        # 计算S-Transform
        S, freqs, times = self.st.fast_transform(
            signal, self.freq_range, self.n_freqs
        )
        S_amp = np.abs(S)
        
        features = {}
        
        # 全局统计特征
        features['st_mean'] = np.mean(S_amp)
        features['st_std'] = np.std(S_amp)
        features['st_max'] = np.max(S_amp)
        features['st_min'] = np.min(S_amp)
        features['st_median'] = np.median(S_amp)
        
        # 各频带的S-Transform特征
        for band_name, (low_freq, high_freq) in self.bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_data = S_amp[band_mask, :]
                features[f'st_{band_name}_mean'] = np.mean(band_data)
                features[f'st_{band_name}_std'] = np.std(band_data)
                features[f'st_{band_name}_max'] = np.max(band_data)
                
                # 时间维度的统计
                band_time_mean = np.mean(band_data, axis=0)
                features[f'st_{band_name}_time_var'] = np.var(band_time_mean)
        
        # 频率维度的特征
        freq_power = np.mean(S_amp, axis=1)  # 对时间取平均
        features['st_freq_centroid'] = np.sum(freqs * freq_power) / np.sum(freq_power) if np.sum(freq_power) > 0 else 0
        features['st_freq_spread'] = np.sqrt(np.sum(((freqs - features['st_freq_centroid']) ** 2) * freq_power) / np.sum(freq_power)) if np.sum(freq_power) > 0 else 0
        
        # 时间维度的特征
        time_power = np.mean(S_amp, axis=0)  # 对频率取平均
        features['st_time_entropy'] = -np.sum(time_power * np.log(time_power + 1e-10)) / len(time_power)
        
        return features
    
    def extract_region_st_features(self, 
                                    data: np.ndarray, 
                                    channel_names: List[str],
                                    region: str) -> Dict[str, float]:
        """
        从特定脑区提取S-Transform特征
        
        Args:
            data: EEG数据 (channels x samples)
            channel_names: 通道名称
            region: 脑区名称
            
        Returns:
            该脑区的S-Transform特征
        """
        region_channels = self.regions[region]
        indices = [i for i, ch in enumerate(channel_names) if ch in region_channels]
        
        if len(indices) == 0:
            return {}
        
        # 对该脑区的所有通道提取特征
        all_features = []
        for idx in indices:
            features = self.extract_st_features(data[idx])
            all_features.append(features)
        
        # 计算平均值
        avg_features = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            avg_features[f'{region}_{key}'] = np.mean(values)
        
        return avg_features
    
    def extract_cnn_features(self, 
                             data: np.ndarray,
                             channel_names: List[str] = None) -> np.ndarray:
        """
        使用CNN提取时频特征
        
        Args:
            data: EEG数据 (channels x samples) 或 (batch x channels x samples)
            channel_names: 通道名称（可选）
            
        Returns:
            CNN特征向量
        """
        # 准备输入
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # 添加batch维度
        
        batch_size, n_channels, n_samples = data.shape
        
        # 计算S-Transform时频图
        tf_images = prepare_stockwell_input(
            data, self.sample_rate, self.freq_range, self.n_freqs
        )
        
        n_times = tf_images.shape[-1]
        
        # 初始化CNN模型（如果需要）
        if self.cnn_model is None:
            self._init_cnn_model(n_channels, n_times)
        
        # 转换为张量
        tf_tensor = torch.FloatTensor(tf_images).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.cnn_model(tf_tensor)
        
        return features.cpu().numpy()
    
    def extract_all_features(self, 
                             data: np.ndarray, 
                             channel_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        提取所有特征并拼接
        
        Args:
            data: EEG数据 (channels x samples)
            channel_names: 通道名称
            
        Returns:
            feature_vector: 拼接后的特征向量
            feature_names: 特征名称列表
        """
        all_features = {}
        
        # 1. 传统特征（RBP、比值、α峰值）
        traditional_features = self.traditional_extractor.extract_all_region_features(
            data, channel_names
        )
        all_features.update(traditional_features)
        
        # 全局传统特征
        global_traditional = self.traditional_extractor.extract_global_features(data)
        all_features.update(global_traditional)
        
        # 前额叶不对称性
        frontal_asymmetry = self.inter_region_extractor.compute_frontal_asymmetry(
            data, channel_names
        )
        all_features.update(frontal_asymmetry)
        
        # 2. S-Transform统计特征
        for region in self.regions.keys():
            try:
                st_features = self.extract_region_st_features(data, channel_names, region)
                all_features.update(st_features)
            except Exception as e:
                print(f"提取 {region} S-Transform特征失败: {e}")
        
        # 3. CNN时频特征
        if self.use_cnn:
            try:
                cnn_features = self.extract_cnn_features(data, channel_names)
                cnn_features = cnn_features.flatten()
                for i, val in enumerate(cnn_features):
                    all_features[f'cnn_feature_{i}'] = val
            except Exception as e:
                print(f"提取CNN特征失败: {e}")
        
        # 转换为数组
        feature_names = list(all_features.keys())
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector, feature_names
    
    def extract_batch_features(self, 
                               data: np.ndarray, 
                               channel_names: List[str],
                               verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取特征
        
        Args:
            data: EEG数据 (batch x channels x samples)
            channel_names: 通道名称
            verbose: 是否显示进度
            
        Returns:
            features: 特征矩阵 (batch x n_features)
            feature_names: 特征名称列表
        """
        n_samples = data.shape[0]
        all_features = []
        feature_names = None
        
        iterator = tqdm(range(n_samples), desc="提取特征") if verbose else range(n_samples)
        
        for i in iterator:
            features, names = self.extract_all_features(data[i], channel_names)
            all_features.append(features)
            
            if feature_names is None:
                feature_names = names
        
        return np.array(all_features), feature_names


def extract_and_save_features(data: np.ndarray,
                               labels: np.ndarray,
                               channel_names: List[str],
                               subject_ids: np.ndarray,
                               output_path: str,
                               sample_rate: float = 500.0,
                               use_cnn: bool = True,
                               device: str = 'cpu'):
    """
    提取所有特征并保存
    
    Args:
        data: EEG数据 (batch x channels x samples)
        labels: 标签
        channel_names: 通道名称
        subject_ids: 受试者ID
        output_path: 输出路径
        sample_rate: 采样率
        use_cnn: 是否使用CNN
        device: 计算设备
    """
    # 创建特征提取器
    extractor = ComprehensiveFeatureExtractor(
        sample_rate=sample_rate,
        use_cnn=use_cnn,
        device=device
    )
    
    # 提取特征
    print("开始提取特征...")
    features, feature_names = extractor.extract_batch_features(data, channel_names)
    
    print(f"特征形状: {features.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'features': features,
        'labels': labels,
        'feature_names': feature_names,
        'subject_ids': subject_ids,
        'channel_names': channel_names
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"特征已保存到: {output_path}")
    
    return features, feature_names


def get_feature_summary(feature_names: List[str]) -> Dict[str, int]:
    """
    获取特征类型统计
    
    Args:
        feature_names: 特征名称列表
        
    Returns:
        各类型特征的数量
    """
    summary = {
        'traditional': 0,
        's_transform': 0,
        'cnn': 0,
        'total': len(feature_names)
    }
    
    for name in feature_names:
        if name.startswith('cnn_'):
            summary['cnn'] += 1
        elif 'st_' in name:
            summary['s_transform'] += 1
        else:
            summary['traditional'] += 1
    
    return summary


if __name__ == "__main__":
    # 测试代码
    import sys
    
    print("=" * 70)
    print("综合特征提取测试")
    print("=" * 70)
    
    # 创建测试数据
    sample_rate = 500.0
    window_size = 4.0
    n_samples = int(sample_rate * window_size)
    n_channels = 19
    batch_size = 3
    
    # 通道名称
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    # 模拟EEG数据
    np.random.seed(42)
    t = np.arange(n_samples) / sample_rate
    
    data = np.zeros((batch_size, n_channels, n_samples))
    for b in range(batch_size):
        for ch in range(n_channels):
            # 添加不同频率成分
            data[b, ch] = (
                2.0 * np.sin(2 * np.pi * 2 * t) +   # delta
                1.5 * np.sin(2 * np.pi * 6 * t) +   # theta
                3.0 * np.sin(2 * np.pi * 10 * t) +  # alpha
                1.0 * np.sin(2 * np.pi * 20 * t) +  # beta
                0.5 * np.random.randn(n_samples)     # 噪声
            )
    
    print(f"\n输入数据形状: {data.shape}")
    print(f"采样率: {sample_rate} Hz")
    print(f"窗口大小: {window_size} 秒")
    print(f"通道数: {n_channels}")
    
    # 创建特征提取器
    print("\n创建特征提取器...")
    extractor = ComprehensiveFeatureExtractor(
        sample_rate=sample_rate,
        use_cnn=True,
        cnn_feature_dim=64,
        device='cpu'
    )
    
    # 测试单样本特征提取
    print("\n" + "-" * 50)
    print("测试单样本特征提取:")
    print("-" * 50)
    
    features, feature_names = extractor.extract_all_features(data[0], channel_names)
    print(f"特征维度: {features.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # 特征类型统计
    summary = get_feature_summary(feature_names)
    print(f"\n特征类型统计:")
    print(f"  传统特征: {summary['traditional']}")
    print(f"  S-Transform特征: {summary['s_transform']}")
    print(f"  CNN特征: {summary['cnn']}")
    print(f"  总计: {summary['total']}")
    
    # 显示部分特征名称
    print(f"\n传统特征示例:")
    traditional_names = [n for n in feature_names if not n.startswith('cnn_') and 'st_' not in n]
    for name in traditional_names[:10]:
        idx = feature_names.index(name)
        print(f"  {name}: {features[idx]:.6f}")
    
    print(f"\nS-Transform特征示例:")
    st_names = [n for n in feature_names if 'st_' in n]
    for name in st_names[:10]:
        idx = feature_names.index(name)
        print(f"  {name}: {features[idx]:.6f}")
    
    # 测试批量特征提取
    print("\n" + "-" * 50)
    print("测试批量特征提取:")
    print("-" * 50)
    
    batch_features, _ = extractor.extract_batch_features(data, channel_names, verbose=True)
    print(f"批量特征形状: {batch_features.shape}")
    
    # 检查NaN
    nan_count = np.sum(np.isnan(batch_features))
    print(f"NaN数量: {nan_count}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


