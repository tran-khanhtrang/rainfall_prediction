import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiDelayDataset(Dataset):
    def __init__(self, csv_path, target_col='ANNUAL', win_len=5, delays=(1,2,3), split='all'):
        df = pd.read_csv(csv_path)
        df = df.sort_values('YEAR').reset_index(drop=True)
        y = df[target_col].astype(float).values
        years = df['YEAR'].values

        cutoff = int(np.floor(0.8 * len(df)))
        if split == 'train':
            idx_range = range(0, cutoff)
        elif split == 'test':
            idx_range = range(cutoff, len(df))
        else:
            idx_range = range(0, len(df))

        self.X, self.Y, self.meta = [], [], []
        max_delay = max(delays)
        for t in idx_range:
            t_target = t + 1
            if t - win_len - max_delay + 1 < 0:
                continue
            if t_target >= len(y):
                break
            frames = []
            window = []
            for d in delays:
                end = t - d + 1
                start = end - win_len + 1
                w = y[start:end+1]
                window.append(w)
            window = np.stack(window, axis=0)[:, None, :]
            frames.append(window)
            X = np.stack(frames, axis=0)
            self.X.append(X.astype(np.float32))
            self.Y.append(y[t_target].astype(np.float32))
            self.meta.append(dict(year_pred=int(years[t_target])))

        self.X = np.stack(self.X, axis=0)
        self.Y = np.array(self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.Y[idx]), self.meta[idx]
