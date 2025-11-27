# -*- coding: utf-8 -*-
"""
S-Transform (Stockwell Transform) 实现
基于论文: Attention-augmented stockwell transform and convolutional neural network 
          framework for electroencephalogram-based multi-class classification of 
          Frontotemporal Dementia (Xie et al., 2025)

S-Transform 公式:
S(f,t) = ∫ x(τ) * h(τ-t, f) * e^(-j2πfτ) dτ

其中窗函数:
h(τ-t, f) = e^(-((τ-t)²f²)/2)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, Optional


class StockwellTransform:
    """
    Stockwell Transform (S-Transform) 实现类
    
    S-Transform 结合了短时傅里叶变换(STFT)和小波变换(WT)的优点，
    提供了同时具有时间和频率分辨率的时频表示。
    """
    
    def __init__(self, sample_rate: float = 500.0):
        """
        初始化 S-Transform
        
        Args:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
    
    def _gaussian_window(self, length: int, freq: float, sigma_factor: float = 1.0) -> np.ndarray:
        """
        生成高斯窗函数
        
        根据论文公式: h(τ-t, f) = e^(-((τ-t)²f²)/2)
        
        Args:
            length: 窗口长度
            freq: 当前频率
            sigma_factor: 高斯窗的缩放因子
            
        Returns:
            高斯窗函数
        """
        if freq == 0:
            return np.ones(length)
        
        # 创建时间向量
        t = np.arange(length) - length // 2
        
        # 计算高斯窗 (按论文公式)
        sigma = sigma_factor / (2 * np.pi * np.abs(freq))
        gaussian = np.exp(-0.5 * (t / (sigma * self.sample_rate)) ** 2)
        
        return gaussian / gaussian.sum()  # 归一化
    
    def transform(self, signal: np.ndarray, 
                  freq_range: Optional[Tuple[float, float]] = None,
                  n_freqs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算信号的 S-Transform
        
        Args:
            signal: 输入信号 (1D array)
            freq_range: 频率范围 (min_freq, max_freq)，默认为 (0.5, 45) Hz
            n_freqs: 频率采样点数，默认自动计算
            
        Returns:
            S: S-Transform 结果矩阵 (频率 x 时间)
            freqs: 频率向量
            times: 时间向量
        """
        n_samples = len(signal)
        
        # 设置默认频率范围
        if freq_range is None:
            freq_range = (0.5, 45.0)
        
        min_freq, max_freq = freq_range
        
        # 计算频率向量
        if n_freqs is None:
            n_freqs = min(int((max_freq - min_freq) * 2) + 1, n_samples // 2)
        
        freqs = np.linspace(min_freq, max_freq, n_freqs)
        
        # 计算时间向量
        times = np.arange(n_samples) / self.sample_rate
        
        # 计算信号的 FFT
        signal_fft = fft(signal)
        
        # 初始化 S-Transform 结果矩阵
        S = np.zeros((n_freqs, n_samples), dtype=np.complex128)
        
        # 对每个频率计算 S-Transform
        for i, freq in enumerate(freqs):
            if freq == 0:
                # 零频率情况：取信号均值
                S[i, :] = np.mean(signal)
            else:
                # 生成高斯窗的频域表示
                gaussian_fft = self._gaussian_window_fft(n_samples, freq)
                
                # 卷积计算 (在频域中进行)
                # 将信号FFT与高斯窗相乘，然后逆FFT
                for j in range(n_samples):
                    # 循环移位
                    shifted_gaussian = np.roll(gaussian_fft, j)
                    # 频域乘法
                    product = signal_fft * shifted_gaussian
                    # 逆FFT得到该时间点的S值
                    S[i, j] = np.sum(product * np.exp(2j * np.pi * freq * np.arange(n_samples) / self.sample_rate)) / n_samples
        
        return S, freqs, times
    
    def _gaussian_window_fft(self, length: int, freq: float) -> np.ndarray:
        """
        计算高斯窗的频域表示
        
        Args:
            length: 信号长度
            freq: 中心频率
            
        Returns:
            高斯窗的FFT
        """
        # 创建频率向量
        fft_freqs = fftfreq(length, 1.0 / self.sample_rate)
        
        # 高斯窗的FFT也是高斯函数
        sigma = np.abs(freq) / (2 * np.pi)
        if sigma == 0:
            return np.ones(length)
        
        gaussian_fft = np.exp(-2 * (np.pi * sigma) ** 2 * (fft_freqs / freq) ** 2)
        
        return gaussian_fft
    
    def fast_transform(self, signal: np.ndarray,
                       freq_range: Optional[Tuple[float, float]] = None,
                       n_freqs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        快速 S-Transform 实现 (使用FFT优化)
        
        这是一个更高效的实现，利用FFT的性质加速计算
        
        Args:
            signal: 输入信号 (1D array)
            freq_range: 频率范围 (min_freq, max_freq)
            n_freqs: 频率采样点数
            
        Returns:
            S: S-Transform 幅值矩阵 (频率 x 时间)
            freqs: 频率向量
            times: 时间向量
        """
        n_samples = len(signal)
        
        # 设置默认频率范围
        if freq_range is None:
            freq_range = (0.5, 45.0)
        
        min_freq, max_freq = freq_range
        
        # 计算频率向量
        if n_freqs is None:
            n_freqs = min(int((max_freq - min_freq) * 2) + 1, 128)
        
        freqs = np.linspace(min_freq, max_freq, n_freqs)
        times = np.arange(n_samples) / self.sample_rate
        
        # 计算信号的 FFT
        signal_fft = fft(signal)
        fft_freqs = fftfreq(n_samples, 1.0 / self.sample_rate)
        
        # 初始化结果矩阵
        S = np.zeros((n_freqs, n_samples), dtype=np.complex128)
        
        for i, freq in enumerate(freqs):
            if freq == 0:
                S[i, :] = np.mean(signal)
            else:
                # 计算高斯窗的频域表示
                # 根据S-Transform理论，窗宽与频率成反比
                sigma_f = np.abs(freq) / (2 * np.pi)
                
                # 高斯窗在频域中的表示
                gauss_fft = np.exp(-((fft_freqs - freq) ** 2) / (2 * sigma_f ** 2))
                
                # 频域乘法后逆FFT
                S[i, :] = ifft(signal_fft * gauss_fft)
        
        return S, freqs, times
    
    def get_time_frequency_matrix(self, signal: np.ndarray,
                                   freq_range: Optional[Tuple[float, float]] = None,
                                   n_freqs: int = 64,
                                   return_amplitude: bool = True) -> np.ndarray:
        """
        获取时频矩阵（用于CNN输入）
        
        Args:
            signal: 输入信号
            freq_range: 频率范围
            n_freqs: 频率采样点数
            return_amplitude: 是否返回幅值（默认True），否则返回复数
            
        Returns:
            时频矩阵
        """
        S, freqs, times = self.fast_transform(signal, freq_range, n_freqs)
        
        if return_amplitude:
            return np.abs(S)
        return S
    
    def extract_band_features(self, signal: np.ndarray,
                               bands: dict = None) -> dict:
        """
        从S-Transform结果中提取频带特征
        
        Args:
            signal: 输入信号
            bands: 频带定义字典，格式为 {band_name: (low_freq, high_freq)}
            
        Returns:
            各频带的特征字典
        """
        if bands is None:
            bands = {
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0)
            }
        
        # 计算完整的S-Transform
        freq_range = (0.5, 45.0)
        S, freqs, times = self.fast_transform(signal, freq_range, n_freqs=128)
        S_amplitude = np.abs(S)
        
        features = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # 找到对应频带的索引
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = S_amplitude[band_mask, :]
            
            # 计算频带特征
            features[f'{band_name}_mean_power'] = np.mean(band_power)
            features[f'{band_name}_max_power'] = np.max(band_power)
            features[f'{band_name}_std_power'] = np.std(band_power)
            features[f'{band_name}_total_power'] = np.sum(band_power)
        
        # 计算总功率用于归一化
        total_power = np.sum(S_amplitude)
        
        # 计算相对功率
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_total = np.sum(S_amplitude[band_mask, :])
            features[f'{band_name}_relative_power'] = band_total / total_power if total_power > 0 else 0
        
        return features


def stockwell_transform_batch(signals: np.ndarray, 
                               sample_rate: float = 500.0,
                               freq_range: Tuple[float, float] = (0.5, 45.0),
                               n_freqs: int = 64) -> np.ndarray:
    """
    批量计算多个信号的S-Transform
    
    Args:
        signals: 输入信号矩阵 (n_signals x n_samples) 或 (n_channels x n_samples)
        sample_rate: 采样率
        freq_range: 频率范围
        n_freqs: 频率采样点数
        
    Returns:
        S-Transform结果 (n_signals x n_freqs x n_samples)
    """
    st = StockwellTransform(sample_rate)
    
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    
    n_signals, n_samples = signals.shape
    result = np.zeros((n_signals, n_freqs, n_samples))
    
    for i in range(n_signals):
        S, _, _ = st.fast_transform(signals[i], freq_range, n_freqs)
        result[i] = np.abs(S)
    
    return result


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 创建测试信号：包含不同频率成分
    sample_rate = 500
    duration = 4.0
    t = np.arange(0, duration, 1/sample_rate)
    
    # 合成信号：5Hz + 10Hz + 20Hz
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    signal += 0.1 * np.random.randn(len(t))  # 添加噪声
    
    # 计算S-Transform
    st = StockwellTransform(sample_rate)
    S, freqs, times = st.fast_transform(signal, freq_range=(0.5, 30), n_freqs=64)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('原始信号')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅值')
    
    plt.subplot(2, 1, 2)
    plt.pcolormesh(times, freqs, np.abs(S), shading='auto', cmap='jet')
    plt.colorbar(label='幅值')
    plt.title('S-Transform 时频图')
    plt.xlabel('时间 (s)')
    plt.ylabel('频率 (Hz)')
    
    plt.tight_layout()
    plt.savefig('s_transform_test.png', dpi=150)
    plt.show()
    
    print("S-Transform 测试完成！")
    print(f"输入信号形状: {signal.shape}")
    print(f"S-Transform输出形状: {S.shape}")


