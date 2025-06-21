# src/processing/data_analysis.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import project modules
from src.ml_pipeline.preprocessing.data_loader import SessionDataLoader
from src.ml_pipeline.preprocessing.preprocessing import process_gsr_signal
from src.ml_pipeline.feature_engineering.feature_engineering import align_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)

class GSRFeatureExtractor:
    """
    Class for extracting advanced features from GSR signals.

    This class provides methods for extracting various features from GSR signals,
    including statistical features, frequency domain features, and non-linear features.
    """

    def __init__(self, sampling_rate: int = 32):
        """
        Initialize the GSR feature extractor.

        Args:
            sampling_rate (int): Sampling rate of the GSR signal in Hz.
        """
        self.sampling_rate = sampling_rate

    def extract_statistical_features(self, gsr_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from a GSR signal.

        Args:
            gsr_signal (np.ndarray): GSR signal time series.

        Returns:
            Dict[str, float]: Dictionary of statistical features.
        """
        features = {}

        # Basic statistics
        features["mean"] = np.mean(gsr_signal)
        features["std"] = np.std(gsr_signal)
        features["min"] = np.min(gsr_signal)
        features["max"] = np.max(gsr_signal)
        features["range"] = features["max"] - features["min"]
        features["median"] = np.median(gsr_signal)

        # Percentiles
        features["percentile_25"] = np.percentile(gsr_signal, 25)
        features["percentile_75"] = np.percentile(gsr_signal, 75)
        features["iqr"] = features["percentile_75"] - features["percentile_25"]

        # Higher-order statistics
        features["skewness"] = self._calculate_skewness(gsr_signal)
        features["kurtosis"] = self._calculate_kurtosis(gsr_signal)

        # Signal dynamics
        features["mean_derivative"] = np.mean(np.diff(gsr_signal))
        features["std_derivative"] = np.std(np.diff(gsr_signal))

        return features

    def extract_frequency_features(self, gsr_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from a GSR signal.

        Args:
            gsr_signal (np.ndarray): GSR signal time series.

        Returns:
            Dict[str, float]: Dictionary of frequency domain features.
        """
        features = {}

        # Compute power spectral density
        f, psd = signal.welch(gsr_signal, fs=self.sampling_rate, nperseg=min(256, len(gsr_signal)))

        # Total power
        features["total_power"] = np.sum(psd)

        # Power in specific frequency bands
        # Very low frequency (VLF): 0.0033-0.04 Hz
        vlf_indices = np.logical_and(f >= 0.0033, f < 0.04)
        features["vlf_power"] = np.sum(psd[vlf_indices])

        # Low frequency (LF): 0.04-0.15 Hz
        lf_indices = np.logical_and(f >= 0.04, f < 0.15)
        features["lf_power"] = np.sum(psd[lf_indices])

        # High frequency (HF): 0.15-0.4 Hz
        hf_indices = np.logical_and(f >= 0.15, f < 0.4)
        features["hf_power"] = np.sum(psd[hf_indices])

        # Relative powers
        if features["total_power"] > 0:
            features["vlf_power_rel"] = features["vlf_power"] / features["total_power"]
            features["lf_power_rel"] = features["lf_power"] / features["total_power"]
            features["hf_power_rel"] = features["hf_power"] / features["total_power"]
        else:
            features["vlf_power_rel"] = 0
            features["lf_power_rel"] = 0
            features["hf_power_rel"] = 0

        # LF/HF ratio
        if features["hf_power"] > 0:
            features["lf_hf_ratio"] = features["lf_power"] / features["hf_power"]
        else:
            features["lf_hf_ratio"] = 0

        # Peak frequency
        if len(psd) > 0:
            features["peak_frequency"] = f[np.argmax(psd)]
        else:
            features["peak_frequency"] = 0

        return features

    def extract_nonlinear_features(self, gsr_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract non-linear features from a GSR signal.

        Args:
            gsr_signal (np.ndarray): GSR signal time series.

        Returns:
            Dict[str, float]: Dictionary of non-linear features.
        """
        features = {}

        # Sample Entropy
        features["sample_entropy"] = self._calculate_sample_entropy(gsr_signal)

        # Detrended Fluctuation Analysis (DFA)
        features["dfa_alpha"] = self._calculate_dfa(gsr_signal)

        # Poincaré plot features
        features["poincare_sd1"], features["poincare_sd2"] = self._calculate_poincare(gsr_signal)

        return features

    def extract_all_features(self, gsr_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all available features from a GSR signal.

        Args:
            gsr_signal (np.ndarray): GSR signal time series.

        Returns:
            Dict[str, float]: Dictionary of all features.
        """
        features = {}

        # Extract all feature types
        statistical_features = self.extract_statistical_features(gsr_signal)
        frequency_features = self.extract_frequency_features(gsr_signal)
        nonlinear_features = self.extract_nonlinear_features(gsr_signal)

        # Combine all features
        features.update(statistical_features)
        features.update(frequency_features)
        features.update(nonlinear_features)

        return features

    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of a signal."""
        n = len(x)
        if n == 0:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        return np.sum(((x - mean) / std) ** 3) / n

    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis of a signal."""
        n = len(x)
        if n == 0:
            return 0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0

        return np.sum(((x - mean) / std) ** 4) / n - 3  # -3 to make normal distribution have kurtosis of 0

    def _calculate_sample_entropy(self, x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate sample entropy of a signal.

        Args:
            x (np.ndarray): Input signal.
            m (int): Embedding dimension.
            r (float): Tolerance (typically 0.1 to 0.25 times the standard deviation of the signal).

        Returns:
            float: Sample entropy value.
        """
        # Ensure the signal is long enough
        if len(x) < 2 * m + 1:
            return 0

        # Normalize the signal
        x = (x - np.mean(x)) / np.std(x)

        # Set tolerance
        r = r * np.std(x)

        # Calculate sample entropy
        try:
            from nolds import sampen
            return sampen(x, emb_dim=m, tolerance=r)
        except ImportError:
            # Simplified implementation if nolds is not available
            N = len(x)

            # Create templates of length m and m+1
            xm = np.array([x[i:i+m] for i in range(N-m+1)])
            xm1 = np.array([x[i:i+m+1] for i in range(N-m)])

            # Count matches
            B = 0
            A = 0

            for i in range(N-m):
                # Count matches for m
                matches_m = np.sum(np.max(np.abs(xm[i] - xm), axis=1) < r)
                B += matches_m - 1  # Exclude self-match

                # Count matches for m+1
                matches_m1 = np.sum(np.max(np.abs(xm1[i] - xm1), axis=1) < r)
                A += matches_m1 - 1  # Exclude self-match

            # Calculate sample entropy
            if B == 0 or A == 0:
                return 0

            return -np.log(A / B)

    def _calculate_dfa(self, x: np.ndarray, scales: Optional[List[int]] = None) -> float:
        """
        Calculate Detrended Fluctuation Analysis (DFA) alpha value.

        Args:
            x (np.ndarray): Input signal.
            scales (List[int], optional): List of scales to use for DFA.

        Returns:
            float: DFA alpha value.
        """
        # Ensure the signal is long enough
        if len(x) < 10:
            return 0

        try:
            from nolds import dfa
            return dfa(x)
        except ImportError:
            # Simplified implementation if nolds is not available
            # This is a very basic implementation and may not be accurate
            # For production use, consider using a specialized library

            # Default scales if not provided
            if scales is None:
                scales = np.logspace(1, np.log10(len(x) // 4), 10).astype(int)
                scales = np.unique(scales)

            # Ensure scales are valid
            scales = scales[scales < len(x) // 4]
            if len(scales) < 2:
                return 0

            # Calculate the profile (cumulative sum of the signal)
            profile = np.cumsum(x - np.mean(x))

            # Calculate fluctuation for each scale
            fluctuations = np.zeros(len(scales))

            for i, scale in enumerate(scales):
                # Number of segments
                n_segments = len(profile) // scale

                # Skip if too few segments
                if n_segments < 1:
                    continue

                # Reshape the profile into segments
                segments = np.array([profile[j*scale:(j+1)*scale] for j in range(n_segments)])

                # Create time array for each segment
                t = np.arange(scale)

                # Calculate local trend for each segment using polynomial fit
                local_trends = np.array([np.polyval(np.polyfit(t, segment, 1), t) for segment in segments])

                # Calculate fluctuation as RMS of detrended segments
                fluctuations[i] = np.sqrt(np.mean(np.var(segments - local_trends, axis=1)))

            # Linear fit in log-log space to find alpha
            valid_idx = fluctuations > 0
            if np.sum(valid_idx) < 2:
                return 0

            log_scales = np.log(scales[valid_idx])
            log_fluct = np.log(fluctuations[valid_idx])

            # Linear regression
            alpha = np.polyfit(log_scales, log_fluct, 1)[0]

            return alpha

    def _calculate_poincare(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Poincaré plot features (SD1 and SD2).

        Args:
            x (np.ndarray): Input signal.

        Returns:
            Tuple[float, float]: SD1 and SD2 values.
        """
        # Ensure the signal is long enough
        if len(x) < 2:
            return 0, 0

        # Create Poincaré plot coordinates
        x1 = x[:-1]
        x2 = x[1:]

        # Calculate SD1 and SD2
        sd1 = np.std(np.subtract(x2, x1) / np.sqrt(2))
        sd2 = np.std(np.add(x2, x1) / np.sqrt(2))

        return sd1, sd2


class PPGFeatureExtractor:
    """
    Class for extracting advanced features from PPG signals.

    This class provides methods for extracting various features from PPG signals,
    including heart rate, heart rate variability, and pulse wave features.
    """

    def __init__(self, sampling_rate: int = 32):
        """
        Initialize the PPG feature extractor.

        Args:
            sampling_rate (int): Sampling rate of the PPG signal in Hz.
        """
        self.sampling_rate = sampling_rate

    def extract_heart_rate(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract heart rate from a PPG signal.

        Args:
            ppg_signal (np.ndarray): PPG signal time series.

        Returns:
            Dict[str, float]: Dictionary with heart rate features.
        """
        features = {}

        # Find peaks in the PPG signal
        peaks, _ = signal.find_peaks(ppg_signal, distance=self.sampling_rate * 0.5)

        if len(peaks) < 2:
            features["heart_rate"] = 0
            features["heart_rate_std"] = 0
            return features

        # Calculate heart rate from peak intervals
        intervals = np.diff(peaks) / self.sampling_rate  # in seconds
        heart_rates = 60 / intervals  # in beats per minute

        # Filter out physiologically impossible heart rates
        valid_hr = (heart_rates >= 40) & (heart_rates <= 200)
        heart_rates = heart_rates[valid_hr]

        if len(heart_rates) < 1:
            features["heart_rate"] = 0
            features["heart_rate_std"] = 0
            return features

        # Calculate mean heart rate and standard deviation
        features["heart_rate"] = np.mean(heart_rates)
        features["heart_rate_std"] = np.std(heart_rates)

        return features

    def extract_hrv_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract heart rate variability features from a PPG signal.

        Args:
            ppg_signal (np.ndarray): PPG signal time series.

        Returns:
            Dict[str, float]: Dictionary with HRV features.
        """
        features = {}

        # Find peaks in the PPG signal
        peaks, _ = signal.find_peaks(ppg_signal, distance=self.sampling_rate * 0.5)

        if len(peaks) < 3:
            features["rmssd"] = 0
            features["sdnn"] = 0
            features["pnn50"] = 0
            return features

        # Calculate RR intervals in milliseconds
        rr_intervals = np.diff(peaks) / self.sampling_rate * 1000

        # Filter out physiologically impossible intervals
        valid_rr = (rr_intervals >= 300) & (rr_intervals <= 2000)
        rr_intervals = rr_intervals[valid_rr]

        if len(rr_intervals) < 2:
            features["rmssd"] = 0
            features["sdnn"] = 0
            features["pnn50"] = 0
            return features

        # Calculate time-domain HRV features

        # SDNN: Standard deviation of NN intervals
        features["sdnn"] = np.std(rr_intervals)

        # RMSSD: Root mean square of successive differences
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        features["rmssd"] = rmssd

        # pNN50: Percentage of successive NN intervals that differ by more than 50 ms
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        features["pnn50"] = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0

        return features

    def extract_pulse_wave_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract pulse wave features from a PPG signal.

        Args:
            ppg_signal (np.ndarray): PPG signal time series.

        Returns:
            Dict[str, float]: Dictionary with pulse wave features.
        """
        features = {}

        # Find peaks in the PPG signal
        peaks, _ = signal.find_peaks(ppg_signal, distance=self.sampling_rate * 0.5)

        if len(peaks) < 2:
            features["pulse_amplitude"] = 0
            features["pulse_width"] = 0
            features["pulse_rise_time"] = 0
            features["pulse_fall_time"] = 0
            return features

        # Calculate pulse amplitude (average peak height)
        features["pulse_amplitude"] = np.mean(ppg_signal[peaks])

        # Calculate average pulse width
        pulse_widths = []
        for i in range(len(peaks) - 1):
            # Find the minimum between two peaks
            start_idx = peaks[i]
            end_idx = peaks[i + 1]

            if end_idx - start_idx < 3:
                continue

            # Find the minimum point between the peaks
            min_idx = start_idx + np.argmin(ppg_signal[start_idx:end_idx])

            # Calculate width at half maximum for the pulse
            half_max = (ppg_signal[peaks[i]] + ppg_signal[min_idx]) / 2

            # Find indices where the signal crosses half_max
            above_half_max = ppg_signal[start_idx:end_idx] > half_max
            transitions = np.diff(above_half_max.astype(int))

            # If we can find both rising and falling edges
            if np.sum(transitions == 1) > 0 and np.sum(transitions == -1) > 0:
                rising_idx = start_idx + np.where(transitions == 1)[0][0]
                falling_idx = start_idx + np.where(transitions == -1)[0][-1]

                # Width in seconds
                width = (falling_idx - rising_idx) / self.sampling_rate
                pulse_widths.append(width)

        features["pulse_width"] = np.mean(pulse_widths) if pulse_widths else 0

        # Calculate average rise and fall times
        rise_times = []
        fall_times = []

        for i in range(len(peaks) - 1):
            # Find the minimum between two peaks
            start_idx = peaks[i]
            end_idx = peaks[i + 1]

            if end_idx - start_idx < 3:
                continue

            # Find the minimum point between the peaks
            min_idx = start_idx + np.argmin(ppg_signal[start_idx:end_idx])

            # Rise time: from minimum to next peak
            if i < len(peaks) - 1:
                rise_time = (peaks[i + 1] - min_idx) / self.sampling_rate
                rise_times.append(rise_time)

            # Fall time: from peak to minimum
            fall_time = (min_idx - peaks[i]) / self.sampling_rate
            fall_times.append(fall_time)

        features["pulse_rise_time"] = np.mean(rise_times) if rise_times else 0
        features["pulse_fall_time"] = np.mean(fall_times) if fall_times else 0

        return features

    def extract_all_features(self, ppg_signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all available features from a PPG signal.

        Args:
            ppg_signal (np.ndarray): PPG signal time series.

        Returns:
            Dict[str, float]: Dictionary of all features.
        """
        features = {}

        # Extract all feature types
        hr_features = self.extract_heart_rate(ppg_signal)
        hrv_features = self.extract_hrv_features(ppg_signal)
        pulse_features = self.extract_pulse_wave_features(ppg_signal)

        # Combine all features
        features.update(hr_features)
        features.update(hrv_features)
        features.update(pulse_features)

        return features


class DataVisualizer:
    """
    Class for visualizing multimodal data from the GSR-RGBT system.

    This class provides methods for creating various visualizations of the
    synchronized data, including time series plots, correlation matrices,
    and feature distributions.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the data visualizer.

        Args:
            output_dir (Path, optional): Directory to save visualizations.
                If None, visualizations will not be saved.
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")

    def plot_time_series(self, df: pd.DataFrame, columns: List[str], 
                         title: str = "Time Series Plot", 
                         figsize: Tuple[int, int] = (12, 8),
                         save_as: Optional[str] = None) -> plt.Figure:
        """
        Create a time series plot of selected columns.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (List[str]): List of column names to plot.
            title (str, optional): Plot title.
            figsize (Tuple[int, int], optional): Figure size.
            save_as (str, optional): Filename to save the plot.
                If None, the plot will not be saved.

        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            logging.warning("No timestamp column found. Using index as x-axis.")
            x = df.index
            xlabel = "Sample Index"
        else:
            x = df["timestamp"]
            xlabel = "Time"

        # Plot each column
        for column in columns:
            if column in df.columns:
                ax.plot(x, df[column], label=column)
            else:
                logging.warning(f"Column '{column}' not found in DataFrame.")

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if save_as is not None and self.output_dir is not None:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved plot to {save_path}")

        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                               title: str = "Correlation Matrix",
                               figsize: Tuple[int, int] = (10, 8),
                               save_as: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation matrix heatmap.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (List[str], optional): List of column names to include.
                If None, all numeric columns will be used.
            title (str, optional): Plot title.
            figsize (Tuple[int, int], optional): Figure size.
            save_as (str, optional): Filename to save the plot.
                If None, the plot will not be saved.

        Returns:
            plt.Figure: The created figure.
        """
        # Select columns
        if columns is None:
            # Use all numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Exclude timestamp if it's numeric
            if "timestamp" in numeric_cols:
                numeric_cols.remove("timestamp")

            columns = numeric_cols
        else:
            # Filter out columns that don't exist
            columns = [col for col in columns if col in df.columns]

        if not columns:
            logging.error("No valid columns to plot.")
            return None

        # Calculate correlation matrix
        corr_matrix = df[columns].corr()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)

        # Set title
        ax.set_title(title)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if save_as is not None and self.output_dir is not None:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved plot to {save_path}")

        return fig

    def plot_feature_distributions(self, df: pd.DataFrame, columns: List[str],
                                  title: str = "Feature Distributions",
                                  figsize: Tuple[int, int] = (12, 10),
                                  save_as: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for selected features.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (List[str]): List of column names to plot.
            title (str, optional): Plot title.
            figsize (Tuple[int, int], optional): Figure size.
            save_as (str, optional): Filename to save the plot.
                If None, the plot will not be saved.

        Returns:
            plt.Figure: The created figure.
        """
        # Filter out columns that don't exist
        columns = [col for col in columns if col in df.columns]

        if not columns:
            logging.error("No valid columns to plot.")
            return None

        # Calculate number of rows and columns for subplots
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes array for easy indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()

        # Plot each feature
        for i, column in enumerate(columns):
            ax = axes[i] if i < len(axes) else axes[-1]

            # Create distribution plot
            sns.histplot(df[column], kde=True, ax=ax)

            # Set title
            ax.set_title(column)

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.7)

        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)

        # Set overall title
        fig.suptitle(title, fontsize=16)

        # Tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save if requested
        if save_as is not None and self.output_dir is not None:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved plot to {save_path}")

        return fig

    def plot_pca_visualization(self, df: pd.DataFrame, columns: List[str],
                              n_components: int = 2,
                              title: str = "PCA Visualization",
                              figsize: Tuple[int, int] = (10, 8),
                              save_as: Optional[str] = None) -> plt.Figure:
        """
        Create a PCA visualization of the data.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            columns (List[str]): List of column names to include in the PCA.
            n_components (int, optional): Number of PCA components to compute.
            title (str, optional): Plot title.
            figsize (Tuple[int, int], optional): Figure size.
            save_as (str, optional): Filename to save the plot.
                If None, the plot will not be saved.

        Returns:
            plt.Figure: The created figure.
        """
        # Filter out columns that don't exist
        columns = [col for col in columns if col in df.columns]

        if not columns:
            logging.error("No valid columns to plot.")
            return None

        # Extract data
        X = df[columns].values

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot PCA results
        if n_components == 2:
            # 2D scatter plot
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

            # Set labels
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

        elif n_components == 3:
            # 3D scatter plot
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)

            # Set labels
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")

        else:
            # For more than 3 components, show explained variance
            ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
            ax.set_xlabel("Principal Component")
            ax.set_ylabel("Explained Variance Ratio")
            ax.set_xticks(range(1, n_components + 1))

        # Set title
        ax.set_title(title)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if save_as is not None and self.output_dir is not None:
            save_path = self.output_dir / save_as
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved plot to {save_path}")

        return fig


class DataAnalyzer:
    """
    Class for analyzing multimodal data from the GSR-RGBT system.

    This class provides methods for loading, processing, and analyzing data
    from recording sessions, including feature extraction, visualization,
    and statistical analysis.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the data analyzer.

        Args:
            output_dir (Path, optional): Directory to save analysis results.
                If None, results will not be saved.
        """
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize feature extractors
        self.gsr_extractor = GSRFeatureExtractor()
        self.ppg_extractor = PPGFeatureExtractor()

        # Initialize visualizer
        self.visualizer = DataVisualizer(output_dir)

    def load_session_data(self, session_path: Path, gsr_sampling_rate: int = 32) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess data from a recording session.

        Args:
            session_path (Path): Path to the session directory.
            gsr_sampling_rate (int, optional): Sampling rate of the GSR signal in Hz.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing the loaded data.
                Keys: 'gsr', 'rgb', 'thermal', 'aligned'
        """
        result = {}

        # Load data
        loader = SessionDataLoader(session_path)

        # Load GSR data
        gsr_df = loader.get_gsr_data()
        if gsr_df is None:
            logging.error(f"Failed to load GSR data from {session_path}")
            return result

        # Preprocess GSR data
        processed_gsr = process_gsr_signal(gsr_df, sampling_rate=gsr_sampling_rate)
        if processed_gsr is None:
            logging.error(f"Failed to process GSR data from {session_path}")
            return result

        result['gsr'] = processed_gsr

        # Load RGB data
        rgb_df = loader.get_rgb_data()
        if rgb_df is not None:
            result['rgb'] = rgb_df

        # Load thermal data
        thermal_df = loader.get_thermal_data()
        if thermal_df is not None:
            result['thermal'] = thermal_df

        # Align signals if both GSR and RGB data are available
        if 'gsr' in result and 'rgb' in result:
            aligned_df = align_signals(result['gsr'], result['rgb'])
            if not aligned_df.empty:
                result['aligned'] = aligned_df

                # If thermal data is also available, align it too
                if 'thermal' in result:
                    aligned_thermal = align_signals(result['gsr'], result['thermal'])
                    if not aligned_thermal.empty:
                        # Merge with the already aligned data
                        common_timestamps = aligned_df["timestamp"].intersection(aligned_thermal["timestamp"])
                        aligned_df_filtered = aligned_df[aligned_df["timestamp"].isin(common_timestamps)]
                        aligned_thermal_filtered = aligned_thermal[aligned_thermal["timestamp"].isin(common_timestamps)]

                        # Now merge them
                        result['aligned'] = pd.merge(aligned_df_filtered, aligned_thermal_filtered, on="timestamp")

        return result

    def extract_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from the loaded data.

        Args:
            data (Dict[str, pd.DataFrame]): Dictionary containing the loaded data.
                Should have keys 'gsr', 'rgb', 'thermal', and/or 'aligned'.

        Returns:
            pd.DataFrame: DataFrame containing the extracted features.
        """
        features = {}

        # Extract features from GSR data
        if 'gsr' in data and 'GSR_Phasic' in data['gsr'].columns:
            gsr_features = self.gsr_extractor.extract_all_features(data['gsr']['GSR_Phasic'].values)
            features.update({f"GSR_{k}": v for k, v in gsr_features.items()})

        # Extract features from PPG data (assuming it's in the aligned DataFrame)
        if 'aligned' in data and any(col.startswith('PPG') for col in data['aligned'].columns):
            ppg_cols = [col for col in data['aligned'].columns if col.startswith('PPG')]
            if ppg_cols:
                ppg_features = self.ppg_extractor.extract_all_features(data['aligned'][ppg_cols[0]].values)
                features.update({f"PPG_{k}": v for k, v in ppg_features.items()})

        # Create DataFrame from features
        features_df = pd.DataFrame([features])

        return features_df

    def analyze_session(self, session_path: Path, gsr_sampling_rate: int = 32,
                       save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Perform a complete analysis of a recording session.

        Args:
            session_path (Path): Path to the session directory.
            gsr_sampling_rate (int, optional): Sampling rate of the GSR signal in Hz.
            save_visualizations (bool, optional): Whether to save visualizations.

        Returns:
            Dict[str, Any]: Dictionary containing the analysis results.
        """
        results = {}

        # Load data
        data = self.load_session_data(session_path, gsr_sampling_rate)
        if not data:
            logging.error(f"Failed to load data from {session_path}")
            return results

        results['data'] = data

        # Extract features
        features = self.extract_features(data)
        results['features'] = features

        # Create visualizations
        if save_visualizations and self.output_dir is not None:
            # Create a subdirectory for this session
            session_name = session_path.name
            session_output_dir = self.output_dir / session_name
            session_output_dir.mkdir(parents=True, exist_ok=True)

            # Update visualizer output directory
            self.visualizer.output_dir = session_output_dir

            # Time series plots
            if 'aligned' in data:
                # Plot GSR signals
                gsr_cols = [col for col in data['aligned'].columns if col.startswith('GSR')]
                if gsr_cols:
                    self.visualizer.plot_time_series(
                        data['aligned'], gsr_cols,
                        title="GSR Signals",
                        save_as="gsr_signals.png"
                    )

                # Plot RGB signals
                rgb_cols = [col for col in data['aligned'].columns if col.startswith('RGB')]
                if rgb_cols and len(rgb_cols) <= 6:  # Limit to 6 columns for readability
                    self.visualizer.plot_time_series(
                        data['aligned'], rgb_cols[:6],
                        title="RGB Signals",
                        save_as="rgb_signals.png"
                    )

                # Plot thermal signals if available
                thermal_cols = [col for col in data['aligned'].columns if col.startswith('THERMAL')]
                if thermal_cols and len(thermal_cols) <= 6:  # Limit to 6 columns for readability
                    self.visualizer.plot_time_series(
                        data['aligned'], thermal_cols[:6],
                        title="Thermal Signals",
                        save_as="thermal_signals.png"
                    )

                # Correlation matrix
                self.visualizer.plot_correlation_matrix(
                    data['aligned'],
                    title="Signal Correlations",
                    save_as="correlation_matrix.png"
                )

                # PCA visualization
                numeric_cols = data['aligned'].select_dtypes(include=np.number).columns.tolist()
                if "timestamp" in numeric_cols:
                    numeric_cols.remove("timestamp")

                if len(numeric_cols) > 2:
                    self.visualizer.plot_pca_visualization(
                        data['aligned'], numeric_cols,
                        title="PCA of Signals",
                        save_as="pca_visualization.png"
                    )

        return results

    def analyze_multiple_sessions(self, session_paths: List[Path], gsr_sampling_rate: int = 32,
                                save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Perform analysis on multiple recording sessions and compare results.

        Args:
            session_paths (List[Path]): List of paths to session directories.
            gsr_sampling_rate (int, optional): Sampling rate of the GSR signal in Hz.
            save_visualizations (bool, optional): Whether to save visualizations.

        Returns:
            Dict[str, Any]: Dictionary containing the analysis results.
        """
        results = {}
        all_features = []

        # Analyze each session
        for session_path in session_paths:
            session_results = self.analyze_session(
                session_path, gsr_sampling_rate, save_visualizations
            )

            if 'features' in session_results:
                # Add session identifier
                session_features = session_results['features'].copy()
                session_features['session_id'] = session_path.name

                # Add to combined features
                all_features.append(session_features)

            # Store individual session results
            results[session_path.name] = session_results

        # Combine features from all sessions
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            results['combined_features'] = combined_features

            # Create visualizations for combined features
            if save_visualizations and self.output_dir is not None:
                # Reset visualizer output directory to main output directory
                self.visualizer.output_dir = self.output_dir

                # Feature distributions
                feature_cols = combined_features.select_dtypes(include=np.number).columns.tolist()
                if feature_cols:
                    self.visualizer.plot_feature_distributions(
                        combined_features, feature_cols,
                        title="Feature Distributions Across Sessions",
                        save_as="feature_distributions.png"
                    )

                # PCA of features
                if len(feature_cols) > 2:
                    self.visualizer.plot_pca_visualization(
                        combined_features, feature_cols,
                        title="PCA of Features Across Sessions",
                        save_as="feature_pca.png"
                    )

        return results


# Example usage
if __name__ == "__main__":
    # Set up output directory
    output_dir = Path("data/analysis_results")

    # Create analyzer
    analyzer = DataAnalyzer(output_dir)

    # Find all session directories
    data_dir = Path("data/recordings")
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("Subject_")]

    if session_dirs:
        # Analyze all sessions
        results = analyzer.analyze_multiple_sessions(session_dirs)

        # Print summary
        print("\nAnalysis Summary:")
        print(f"Analyzed {len(session_dirs)} sessions")

        if 'combined_features' in results:
            print("\nExtracted Features:")
            print(results['combined_features'].describe())
    else:
        print("No session directories found.")
