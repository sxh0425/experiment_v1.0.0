# -*- coding: utf-8 -*-
"""
传统EEG特征提取模块

包含以下特征：
1. 相对频带功率 (Relative Band Power, RBP)
2. 频带比值特征 (θ/α, θ/β, (δ+θ)/(α+β))
3. α峰值频率特征

按四个脑区分别提取：前额叶、颞叶、顶叶、枕叶
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# 频带定义
FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)  # 可选
}

# 脑区定义
BRAIN_REGIONS = {
    'frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'parietal': ['P3', 'Pz', 'P4'],
    'occipital': ['O1', 'O2']
}


class TraditionalFeatureExtractor:
    """
    传统EEG特征提取器
    
    提取以下特征：
    - 各频带的相对功率 (RBP)
    - 频带比值特征
    - α峰值特征
    """
    
    def __init__(self, 
                 sample_rate: float = 500.0,
                 bands: Dict[str, Tuple[float, float]] = None,
                 regions: Dict[str, List[str]] = None):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率 (Hz)
            bands: 频带定义
            regions: 脑区定义
        """
        self.sample_rate = sample_rate
        self.bands = bands or FREQUENCY_BANDS
        self.regions = regions or BRAIN_REGIONS
        
        # Welch方法参数
        self.nperseg = int(sample_rate * 2)  # 2秒窗口
        self.noverlap = int(self.nperseg * 0.5)  # 50%重叠
    
    def compute_psd(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度 (PSD)
        
        使用Welch方法
        
        Args:
            signal_data: 输入信号 (1D)
            
        Returns:
            freqs: 频率向量
            psd: 功率谱密度
        """
        freqs, psd = signal.welch(
            signal_data,
            fs=self.sample_rate,
            nperseg=min(self.nperseg, len(signal_data)),
            noverlap=min(self.noverlap, len(signal_data) // 2),
            scaling='density'
        )
        
        return freqs, psd
    
    def compute_band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                          band: Tuple[float, float]) -> float:
        """
        计算特定频带的功率
        
        Args:
            freqs: 频率向量
            psd: 功率谱密度
            band: 频带范围 (low, high)
            
        Returns:
            频带功率
        """
        low, high = band
        mask = (freqs >= low) & (freqs <= high)
        
        if not np.any(mask):
            return 0.0
        
        # 使用梯形积分计算功率
        band_power = np.trapz(psd[mask], freqs[mask])
        
        return band_power
    
    def compute_relative_band_power(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        计算各频带的相对功率 (RBP)
        
        RBP = 特定频带功率 / 总功率
        
        Args:
            signal_data: 输入信号 (1D)
            
        Returns:
            各频带的相对功率字典
        """
        freqs, psd = self.compute_psd(signal_data)
        
        # 计算各频带功率
        band_powers = {}
        for band_name, band_range in self.bands.items():
            band_powers[band_name] = self.compute_band_power(freqs, psd, band_range)
        
        # 计算总功率 (0.5-45Hz)
        total_power = self.compute_band_power(freqs, psd, (0.5, 45.0))
        
        # 计算相对功率
        rbp = {}
        for band_name, power in band_powers.items():
            rbp[f'rbp_{band_name}'] = power / total_power if total_power > 0 else 0.0
        
        return rbp
    
    def compute_ratio_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        计算频带比值特征
        
        包括：
        - θ/α
        - θ/β
        - (δ+θ)/(α+β)
        
        Args:
            signal_data: 输入信号 (1D)
            
        Returns:
            比值特征字典
        """
        freqs, psd = self.compute_psd(signal_data)
        
        # 计算各频带功率
        delta_power = self.compute_band_power(freqs, psd, self.bands['delta'])
        theta_power = self.compute_band_power(freqs, psd, self.bands['theta'])
        alpha_power = self.compute_band_power(freqs, psd, self.bands['alpha'])
        beta_power = self.compute_band_power(freqs, psd, self.bands['beta'])
        
        # 计算比值特征
        eps = 1e-10  # 防止除零
        
        ratios = {
            'ratio_theta_alpha': theta_power / (alpha_power + eps),
            'ratio_theta_beta': theta_power / (beta_power + eps),
            'ratio_slow_fast': (delta_power + theta_power) / (alpha_power + beta_power + eps)
        }
        
        return ratios
    
    def compute_alpha_peak_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        计算α峰值特征
        
        包括：
        - α峰值频率 (PAF - Peak Alpha Frequency)
        - α峰值功率
        - α峰值带宽
        
        Args:
            signal_data: 输入信号 (1D)
            
        Returns:
            α峰值特征字典
        """
        freqs, psd = self.compute_psd(signal_data)
        
        # 获取α频带范围
        alpha_low, alpha_high = self.bands['alpha']
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
        
        # α峰值频率 (最大功率对应的频率)
        peak_idx = np.argmax(alpha_psd)
        alpha_peak_freq = alpha_freqs[peak_idx]
        alpha_peak_power = alpha_psd[peak_idx]
        
        # α中心频率 (功率加权平均)
        total_alpha_power = np.sum(alpha_psd)
        if total_alpha_power > 0:
            alpha_center_freq = np.sum(alpha_freqs * alpha_psd) / total_alpha_power
        else:
            alpha_center_freq = 0.0
        
        # α峰值带宽 (半高全宽 FWHM)
        half_max = alpha_peak_power / 2
        above_half = alpha_psd >= half_max
        if np.any(above_half):
            above_indices = np.where(above_half)[0]
            bandwidth = alpha_freqs[above_indices[-1]] - alpha_freqs[above_indices[0]]
        else:
            bandwidth = 0.0
        
        return {
            'alpha_peak_freq': alpha_peak_freq,
            'alpha_peak_power': alpha_peak_power,
            'alpha_peak_bandwidth': bandwidth,
            'alpha_center_freq': alpha_center_freq
        }
    
    def extract_channel_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        从单个通道提取所有传统特征
        
        Args:
            signal_data: 单通道信号 (1D)
            
        Returns:
            特征字典
        """
        features = {}
        
        # 相对频带功率
        rbp = self.compute_relative_band_power(signal_data)
        features.update(rbp)
        
        # 比值特征
        ratios = self.compute_ratio_features(signal_data)
        features.update(ratios)
        
        # α峰值特征
        alpha_peaks = self.compute_alpha_peak_features(signal_data)
        features.update(alpha_peaks)
        
        return features
    
    def extract_region_features(self, 
                                 data: np.ndarray, 
                                 channel_names: List[str],
                                 region: str) -> Dict[str, float]:
        """
        从特定脑区提取特征（对该区域所有通道取平均）
        
        Args:
            data: EEG数据 (channels x samples)
            channel_names: 通道名称列表
            region: 脑区名称
            
        Returns:
            脑区特征字典
        """
        if region not in self.regions:
            raise ValueError(f"未知脑区: {region}")
        
        region_channels = self.regions[region]
        
        # 获取该脑区的通道索引
        indices = [i for i, ch in enumerate(channel_names) if ch in region_channels]
        
        if len(indices) == 0:
            raise ValueError(f"在数据中找不到脑区 {region} 的通道")
        
        # 提取每个通道的特征，然后取平均
        all_features = []
        for idx in indices:
            channel_features = self.extract_channel_features(data[idx])
            all_features.append(channel_features)
        
        # 计算平均特征
        avg_features = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            avg_features[f'{region}_{key}'] = np.mean(values)
            avg_features[f'{region}_{key}_std'] = np.std(values)
        
        return avg_features
    
    def extract_all_region_features(self, 
                                     data: np.ndarray, 
                                     channel_names: List[str]) -> Dict[str, float]:
        """
        从所有脑区提取特征
        
        Args:
            data: EEG数据 (channels x samples)
            channel_names: 通道名称列表
            
        Returns:
            所有脑区的特征字典
        """
        all_features = {}
        
        for region in self.regions.keys():
            try:
                region_features = self.extract_region_features(data, channel_names, region)
                all_features.update(region_features)
            except Exception as e:
                print(f"提取 {region} 特征失败: {e}")
        
        return all_features
    
    def extract_global_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        从所有通道提取全局特征
        
        Args:
            data: EEG数据 (channels x samples)
            
        Returns:
            全局特征字典
        """
        # 对所有通道取平均
        global_signal = np.mean(data, axis=0)
        
        features = {}
        
        # 全局相对频带功率
        rbp = self.compute_relative_band_power(global_signal)
        for k, v in rbp.items():
            features[f'global_{k}'] = v
        
        # 全局比值特征
        ratios = self.compute_ratio_features(global_signal)
        for k, v in ratios.items():
            features[f'global_{k}'] = v
        
        # 全局α峰值特征
        alpha_peaks = self.compute_alpha_peak_features(global_signal)
        for k, v in alpha_peaks.items():
            features[f'global_{k}'] = v
        
        return features


class InterRegionFeatures:
    """
    脑区间特征提取器
    
    计算不同脑区之间的特征差异和连接性
    """
    
    def __init__(self, 
                 sample_rate: float = 500.0,
                 bands: Dict[str, Tuple[float, float]] = None):
        """
        初始化
        
        Args:
            sample_rate: 采样率
            bands: 频带定义
        """
        self.sample_rate = sample_rate
        self.bands = bands or FREQUENCY_BANDS
        self.extractor = TraditionalFeatureExtractor(sample_rate, bands)
    
    def compute_asymmetry(self, 
                          data: np.ndarray, 
                          channel_names: List[str],
                          left_channels: List[str],
                          right_channels: List[str]) -> Dict[str, float]:
        """
        计算左右半球不对称性
        
        Args:
            data: EEG数据
            channel_names: 通道名称
            left_channels: 左半球通道
            right_channels: 右半球通道
            
        Returns:
            不对称性特征
        """
        # 获取左右半球通道索引
        left_indices = [i for i, ch in enumerate(channel_names) if ch in left_channels]
        right_indices = [i for i, ch in enumerate(channel_names) if ch in right_channels]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return {}
        
        # 计算左右半球平均信号
        left_signal = np.mean(data[left_indices], axis=0)
        right_signal = np.mean(data[right_indices], axis=0)
        
        # 计算各频带的不对称性
        left_rbp = self.extractor.compute_relative_band_power(left_signal)
        right_rbp = self.extractor.compute_relative_band_power(right_signal)
        
        asymmetry = {}
        for band_name in self.bands.keys():
            left_power = left_rbp.get(f'rbp_{band_name}', 0)
            right_power = right_rbp.get(f'rbp_{band_name}', 0)
            
            # 不对称性指数: (R - L) / (R + L)
            total = left_power + right_power
            if total > 0:
                asymmetry[f'asymmetry_{band_name}'] = (right_power - left_power) / total
            else:
                asymmetry[f'asymmetry_{band_name}'] = 0.0
        
        return asymmetry
    
    def compute_frontal_asymmetry(self, 
                                   data: np.ndarray, 
                                   channel_names: List[str]) -> Dict[str, float]:
        """
        计算前额叶不对称性 (常用于情绪/认知研究)
        
        Args:
            data: EEG数据
            channel_names: 通道名称
            
        Returns:
            前额叶不对称性特征
        """
        left_frontal = ['Fp1', 'F3', 'F7']
        right_frontal = ['Fp2', 'F4', 'F8']
        
        asymmetry = self.compute_asymmetry(data, channel_names, left_frontal, right_frontal)
        
        # 重命名为前额叶特定
        frontal_asymmetry = {}
        for k, v in asymmetry.items():
            frontal_asymmetry[f'frontal_{k}'] = v
        
        return frontal_asymmetry


def extract_traditional_features(data: np.ndarray, 
                                  channel_names: List[str],
                                  sample_rate: float = 500.0) -> np.ndarray:
    """
    提取所有传统特征的便捷函数
    
    Args:
        data: EEG数据 (channels x samples) 或 (n_windows x channels x samples)
        channel_names: 通道名称
        sample_rate: 采样率
        
    Returns:
        特征矩阵
    """
    extractor = TraditionalFeatureExtractor(sample_rate)
    inter_region = InterRegionFeatures(sample_rate)
    
    if data.ndim == 2:
        # 单个窗口
        features = {}
        
        # 各脑区特征
        region_features = extractor.extract_all_region_features(data, channel_names)
        features.update(region_features)
        
        # 全局特征
        global_features = extractor.extract_global_features(data)
        features.update(global_features)
        
        # 前额叶不对称性
        frontal_asymmetry = inter_region.compute_frontal_asymmetry(data, channel_names)
        features.update(frontal_asymmetry)
        
        return np.array(list(features.values())), list(features.keys())
    
    elif data.ndim == 3:
        # 多个窗口
        n_windows = data.shape[0]
        all_features = []
        feature_names = None
        
        for i in range(n_windows):
            features_array, names = extract_traditional_features(
                data[i], channel_names, sample_rate
            )
            all_features.append(features_array)
            
            if feature_names is None:
                feature_names = names
        
        return np.array(all_features), feature_names
    
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")


def get_feature_names() -> List[str]:
    """
    获取所有传统特征的名称
    
    Returns:
        特征名称列表
    """
    # 基础特征名称
    base_features = []
    
    # RBP特征
    for band in FREQUENCY_BANDS.keys():
        base_features.append(f'rbp_{band}')
    
    # 比值特征
    base_features.extend(['ratio_theta_alpha', 'ratio_theta_beta', 'ratio_slow_fast'])
    
    # α峰值特征
    base_features.extend(['alpha_peak_freq', 'alpha_peak_power', 
                          'alpha_peak_bandwidth', 'alpha_center_freq'])
    
    # 脑区特征
    all_features = []
    for region in BRAIN_REGIONS.keys():
        for feat in base_features:
            all_features.append(f'{region}_{feat}')
            all_features.append(f'{region}_{feat}_std')
    
    # 全局特征
    for feat in base_features:
        all_features.append(f'global_{feat}')
    
    # 前额叶不对称性
    for band in FREQUENCY_BANDS.keys():
        all_features.append(f'frontal_asymmetry_{band}')
    
    return all_features


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试信号
    sample_rate = 500
    duration = 4.0
    n_samples = int(sample_rate * duration)
    n_channels = 19
    
    t = np.arange(n_samples) / sample_rate
    
    # 模拟多通道EEG信号
    np.random.seed(42)
    data = np.zeros((n_channels, n_samples))
    
    for i in range(n_channels):
        # 添加不同频率成分
        data[i] = (
            2.0 * np.sin(2 * np.pi * 2 * t) +   # delta
            1.5 * np.sin(2 * np.pi * 6 * t) +   # theta
            3.0 * np.sin(2 * np.pi * 10 * t) +  # alpha
            1.0 * np.sin(2 * np.pi * 20 * t) +  # beta
            0.5 * np.random.randn(n_samples)     # 噪声
        )
    
    # 通道名称
    channel_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    # 创建特征提取器
    extractor = TraditionalFeatureExtractor(sample_rate)
    
    print("=" * 60)
    print("传统特征提取测试")
    print("=" * 60)
    
    # 测试单通道特征提取
    print("\n单通道特征提取 (Fp1):")
    channel_features = extractor.extract_channel_features(data[0])
    for name, value in channel_features.items():
        print(f"  {name}: {value:.4f}")
    
    # 测试脑区特征提取
    print("\n脑区特征提取:")
    for region in BRAIN_REGIONS.keys():
        region_features = extractor.extract_region_features(data, channel_names, region)
        print(f"\n  {region}区域:")
        for name, value in list(region_features.items())[:5]:  # 只显示前5个
            print(f"    {name}: {value:.4f}")
    
    # 测试完整特征提取
    print("\n\n完整特征提取:")
    features_array, feature_names = extract_traditional_features(data, channel_names, sample_rate)
    print(f"  特征数量: {len(features_array)}")
    print(f"  特征名称示例: {feature_names[:10]}")
    
    # 可视化PSD
    print("\n\n生成PSD可视化...")
    freqs, psd = extractor.compute_psd(data[0])
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, data[0])
    plt.title('原始信号 (Fp1)')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅值')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(freqs, psd)
    plt.title('功率谱密度 (PSD)')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.xlim([0, 50])
    plt.grid(True)
    
    # 标注频带
    colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'red', 'beta': 'orange'}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        if band_name != 'gamma':
            plt.axvspan(low, high, alpha=0.2, color=colors[band_name], label=band_name)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('traditional_features_test.png', dpi=150)
    print("保存图像到: traditional_features_test.png")
    
    print("\n测试完成！")


