# -*- coding: utf-8 -*-
"""
子窗口特征提取模块

将4秒EEG窗口分成多个子窗口，保留时间维度
输出形状: (T, R, F) - 时间步 × 脑区 × 特征

特征类型：
- RBP: 4个频带的相对功率
- 比值: θ/α, θ/β, (δ+θ)/(α+β)
- α峰值: 峰值频率、功率、带宽、中心频率
- S-Transform统计: 均值、标准差、最大值等
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


# 频带定义
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0)
}

# 脑区定义
BRAIN_REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'parietal': ['P3', 'Pz', 'P4'],
    'occipital': ['O1', 'O2']
}

REGION_ORDER = ['frontal', 'temporal', 'parietal', 'occipital']


class SubwindowFeatureExtractor:
    """
    子窗口特征提取器
    
    将EEG信号分成多个子窗口，每个子窗口提取特征，
    保留时间维度信息
    """
    
    def __init__(self,
                 sample_rate: float = 500.0,
                 window_size: float = 4.0,
                 subwindow_size: float = 0.5,
                 subwindow_overlap: float = 0.5,
                 freq_range: Tuple[float, float] = (0.5, 45.0)):
        """
        初始化
        
        Args:
            sample_rate: 采样率 (Hz)
            window_size: 总窗口大小 (秒)
            subwindow_size: 子窗口大小 (秒)
            subwindow_overlap: 子窗口重叠率
            freq_range: 频率范围
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.subwindow_size = subwindow_size
        self.subwindow_overlap = subwindow_overlap
        self.freq_range = freq_range
        
        # 计算子窗口参数
        self.subwindow_samples = int(subwindow_size * sample_rate)
        self.subwindow_step = int(self.subwindow_samples * (1 - subwindow_overlap))
        self.total_samples = int(window_size * sample_rate)
        
        # 计算时间步数
        self.n_timesteps = (self.total_samples - self.subwindow_samples) // self.subwindow_step + 1
        
        # Welch方法参数
        self.nperseg = min(256, self.subwindow_samples)
        self.noverlap = self.nperseg // 2
        
        # 特征名称
        self.feature_names = self._get_feature_names()
        self.n_features = len(self.feature_names)
        self.n_regions = len(REGION_ORDER)
    
    def _get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        # RBP特征 (4个)
        for band in FREQUENCY_BANDS.keys():
            names.append(f'rbp_{band}')
        
        # 比值特征 (3个)
        names.extend(['ratio_theta_alpha', 'ratio_theta_beta', 'ratio_slow_fast'])
        
        # α峰值特征 (4个)
        names.extend(['alpha_peak_freq', 'alpha_peak_power', 
                      'alpha_peak_bandwidth', 'alpha_center_freq'])
        
        # S-Transform统计特征 (15个)
        names.extend([
            'st_mean', 'st_std', 'st_max', 'st_min', 'st_median',
            'st_delta_mean', 'st_theta_mean', 'st_alpha_mean', 'st_beta_mean',
            'st_delta_std', 'st_theta_std', 'st_alpha_std', 'st_beta_std',
            'st_freq_centroid', 'st_time_entropy'
        ])
        
        return names
    
    def compute_psd(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算功率谱密度"""
        freqs, psd = signal.welch(
            signal_data,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling='density'
        )
        return freqs, psd
    
    def compute_band_power(self, freqs: np.ndarray, psd: np.ndarray,
                           band: Tuple[float, float]) -> float:
        """计算特定频带功率"""
        low, high = band
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return np.trapz(psd[mask], freqs[mask])
    
    def extract_rbp_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取相对频带功率特征"""
        freqs, psd = self.compute_psd(signal_data)
        
        # 计算各频带功率
        band_powers = {}
        for band_name, band_range in FREQUENCY_BANDS.items():
            band_powers[band_name] = self.compute_band_power(freqs, psd, band_range)
        
        # 总功率
        total_power = self.compute_band_power(freqs, psd, (0.5, 45.0))
        
        # 相对功率
        features = {}
        for band_name, power in band_powers.items():
            features[f'rbp_{band_name}'] = power / total_power if total_power > 0 else 0.0
        
        return features
    
    def extract_ratio_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取频带比值特征"""
        freqs, psd = self.compute_psd(signal_data)
        
        delta = self.compute_band_power(freqs, psd, FREQUENCY_BANDS['delta'])
        theta = self.compute_band_power(freqs, psd, FREQUENCY_BANDS['theta'])
        alpha = self.compute_band_power(freqs, psd, FREQUENCY_BANDS['alpha'])
        beta = self.compute_band_power(freqs, psd, FREQUENCY_BANDS['beta'])
        
        eps = 1e-10
        
        return {
            'ratio_theta_alpha': theta / (alpha + eps),
            'ratio_theta_beta': theta / (beta + eps),
            'ratio_slow_fast': (delta + theta) / (alpha + beta + eps)
        }
    
    def extract_alpha_peak_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取α峰值特征"""
        freqs, psd = self.compute_psd(signal_data)
        
        # α频带
        alpha_low, alpha_high = FREQUENCY_BANDS['alpha']
        alpha_mask = (freqs >= alpha_low) & (freqs <= alpha_high)
        
        if not np.any(alpha_mask):
            return {
                'alpha_peak_freq': 0.0,
                'alpha_peak_power': 0.0,
                'alpha_peak_bandwidth': 0.0,
                'alpha_center_freq': 0.0
            }
        
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = psd[alpha_mask]
        
        # 峰值频率和功率
        peak_idx = np.argmax(alpha_psd)
        alpha_peak_freq = alpha_freqs[peak_idx]
        alpha_peak_power = alpha_psd[peak_idx]
        
        # 中心频率
        total_alpha = np.sum(alpha_psd)
        alpha_center_freq = np.sum(alpha_freqs * alpha_psd) / total_alpha if total_alpha > 0 else 0.0
        
        # 带宽 (FWHM)
        half_max = alpha_peak_power / 2
        above_half = alpha_psd >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            bandwidth = alpha_freqs[indices[-1]] - alpha_freqs[indices[0]]
        else:
            bandwidth = 0.0
        
        return {
            'alpha_peak_freq': alpha_peak_freq,
            'alpha_peak_power': alpha_peak_power,
            'alpha_peak_bandwidth': bandwidth,
            'alpha_center_freq': alpha_center_freq
        }
    
    def extract_st_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """提取S-Transform统计特征（简化版）"""
        # 使用FFT近似S-Transform的频带统计
        n = len(signal_data)
        freqs = fftfreq(n, 1.0 / self.sample_rate)
        fft_vals = np.abs(fft(signal_data))
        
        # 只取正频率
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # 限制在感兴趣的频率范围
        range_mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        freqs = freqs[range_mask]
        fft_vals = fft_vals[range_mask]
        
        if len(fft_vals) == 0:
            return {name: 0.0 for name in self.feature_names if name.startswith('st_')}
        
        # 全局统计
        features = {
            'st_mean': np.mean(fft_vals),
            'st_std': np.std(fft_vals),
            'st_max': np.max(fft_vals),
            'st_min': np.min(fft_vals),
            'st_median': np.median(fft_vals)
        }
        
        # 各频带统计
        for band_name, (low, high) in FREQUENCY_BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_vals = fft_vals[band_mask]
                features[f'st_{band_name}_mean'] = np.mean(band_vals)
                features[f'st_{band_name}_std'] = np.std(band_vals)
            else:
                features[f'st_{band_name}_mean'] = 0.0
                features[f'st_{band_name}_std'] = 0.0
        
        # 频率质心
        total = np.sum(fft_vals)
        features['st_freq_centroid'] = np.sum(freqs * fft_vals) / total if total > 0 else 0.0
        
        # 时间熵（使用归一化的幅度分布）
        normalized = fft_vals / (total + 1e-10)
        features['st_time_entropy'] = -np.sum(normalized * np.log(normalized + 1e-10))
        
        return features
    
    def extract_subwindow_features(self, signal_data: np.ndarray) -> np.ndarray:
        """
        提取单个通道单个子窗口的所有特征
        
        Args:
            signal_data: 子窗口信号 (1D)
            
        Returns:
            特征向量 (n_features,)
        """
        features = {}
        
        # RBP特征
        features.update(self.extract_rbp_features(signal_data))
        
        # 比值特征
        features.update(self.extract_ratio_features(signal_data))
        
        # α峰值特征
        features.update(self.extract_alpha_peak_features(signal_data))
        
        # S-Transform特征
        features.update(self.extract_st_features(signal_data))
        
        # 按顺序返回
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def extract_region_features(self, 
                                 data: np.ndarray,
                                 channel_names: List[str],
                                 region: str,
                                 start_idx: int) -> np.ndarray:
        """
        提取特定脑区在特定子窗口的特征
        
        Args:
            data: EEG数据 (channels, samples)
            channel_names: 通道名称
            region: 脑区名称
            start_idx: 子窗口起始索引
            
        Returns:
            该脑区的特征向量 (n_features,)
        """
        region_channels = BRAIN_REGIONS[region]
        indices = [i for i, ch in enumerate(channel_names) if ch in region_channels]
        
        if len(indices) == 0:
            return np.zeros(self.n_features)
        
        # 提取子窗口数据
        end_idx = start_idx + self.subwindow_samples
        subwindow_data = data[indices, start_idx:end_idx]
        
        # 对该脑区的所有通道提取特征，然后取平均
        all_features = []
        for ch_idx in range(len(indices)):
            features = self.extract_subwindow_features(subwindow_data[ch_idx])
            all_features.append(features)
        
        return np.mean(all_features, axis=0)
    
    def extract_all_features(self,
                             data: np.ndarray,
                             channel_names: List[str]) -> np.ndarray:
        """
        提取完整的特征张量
        
        Args:
            data: EEG数据 (channels, samples) - 单个4秒窗口
            channel_names: 通道名称列表
            
        Returns:
            特征张量 (T, R, F) - 时间步×脑区×特征
        """
        # 初始化特征张量
        feature_tensor = np.zeros((self.n_timesteps, self.n_regions, self.n_features))
        
        # 遍历每个时间步
        for t in range(self.n_timesteps):
            start_idx = t * self.subwindow_step
            
            # 遍历每个脑区
            for r, region in enumerate(REGION_ORDER):
                features = self.extract_region_features(
                    data, channel_names, region, start_idx
                )
                feature_tensor[t, r, :] = features
        
        return feature_tensor
    
    def extract_batch_features(self,
                               data: np.ndarray,
                               channel_names: List[str],
                               verbose: bool = True) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            data: EEG数据 (batch, channels, samples)
            channel_names: 通道名称
            verbose: 是否显示进度
            
        Returns:
            特征张量 (batch, T, R, F)
        """
        n_samples = data.shape[0]
        feature_tensors = np.zeros((n_samples, self.n_timesteps, 
                                    self.n_regions, self.n_features))
        
        iterator = tqdm(range(n_samples), desc="提取子窗口特征") if verbose else range(n_samples)
        
        for i in iterator:
            feature_tensors[i] = self.extract_all_features(data[i], channel_names)
        
        return feature_tensors
    
    def get_info(self) -> Dict:
        """获取提取器信息"""
        return {
            'sample_rate': self.sample_rate,
            'window_size': self.window_size,
            'subwindow_size': self.subwindow_size,
            'subwindow_overlap': self.subwindow_overlap,
            'n_timesteps': self.n_timesteps,
            'n_regions': self.n_regions,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'region_order': REGION_ORDER,
            'output_shape': (self.n_timesteps, self.n_regions, self.n_features)
        }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("子窗口特征提取测试")
    print("=" * 60)
    
    # 创建测试数据
    sample_rate = 500.0
    window_size = 4.0
    n_samples = int(sample_rate * window_size)
    n_channels = 19
    
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    # 模拟EEG数据
    np.random.seed(42)
    t = np.arange(n_samples) / sample_rate
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = (
            2.0 * np.sin(2 * np.pi * 2 * t) +   # delta
            1.5 * np.sin(2 * np.pi * 6 * t) +   # theta
            3.0 * np.sin(2 * np.pi * 10 * t) +  # alpha
            1.0 * np.sin(2 * np.pi * 20 * t) +  # beta
            0.5 * np.random.randn(n_samples)
        )
    
    # 创建提取器
    extractor = SubwindowFeatureExtractor(
        sample_rate=sample_rate,
        window_size=window_size,
        subwindow_size=0.5,
        subwindow_overlap=0.5
    )
    
    # 打印信息
    info = extractor.get_info()
    print(f"\n提取器配置:")
    print(f"  采样率: {info['sample_rate']} Hz")
    print(f"  窗口大小: {info['window_size']} 秒")
    print(f"  子窗口大小: {info['subwindow_size']} 秒")
    print(f"  子窗口重叠: {info['subwindow_overlap']}")
    print(f"  时间步数(T): {info['n_timesteps']}")
    print(f"  脑区数(R): {info['n_regions']}")
    print(f"  特征数(F): {info['n_features']}")
    print(f"  输出形状: {info['output_shape']}")
    print(f"\n特征名称: {info['feature_names']}")
    
    # 提取特征
    print(f"\n提取单个样本特征...")
    feature_tensor = extractor.extract_all_features(data, channel_names)
    print(f"特征张量形状: {feature_tensor.shape}")
    
    # 检查NaN
    nan_count = np.sum(np.isnan(feature_tensor))
    print(f"NaN数量: {nan_count}")
    
    # 批量测试
    print(f"\n批量提取测试...")
    batch_data = np.stack([data] * 5)  # 5个样本
    batch_features = extractor.extract_batch_features(batch_data, channel_names)
    print(f"批量特征形状: {batch_features.shape}")
    
    print("\n测试完成！")

