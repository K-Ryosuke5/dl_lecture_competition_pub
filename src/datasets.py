import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import scipy.signal

# フィルタの設定
filter_type = 'lowpass'  # ローパスフィルタを使用する場合
cutoff_freq = 30  # カットオフ周波数
sampling_freq = 100  # サンプリング周波数
filter_order = 4  # フィルタ次数

# フィルタ係数の計算
filter_coefficients = scipy.signal.butter(filter_order, cutoff_freq, btype=filter_type, fs=sampling_freq, output='sos')

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", resample_rate: int = 100) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        

        
        # データのリサンプリング
        self.X = self.resample_data(self.X, resample_rate)
        
        # データのフィルタリング
        #self.X = self.filter_data(self.X)



    def resample_data(self, data, resample_rate):
        # データのリサンプリング処理を実装
        # 例えば、scipyのresample関数を使用してデータをリサンプリングする
        resampled_data = scipy.signal.resample(data, resample_rate, axis=2)
        return resampled_data.copy()  # 負のストライドを避けるためにコピーを作成

    def filter_data(self, data):
        # データのフィルタリング処理を実装
        # 例えば、scipyのフィルタリング関数を使用してデータをフィルタリングする
        filtered_data = scipy.signal.sosfiltfilt(filter_coefficients, data, axis=2)
        return filtered_data.copy()  # 負のストライドを避けるためにコピーを作成

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]