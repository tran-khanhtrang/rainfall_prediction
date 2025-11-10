# predict_from_ckpt.py
import json
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from convlstm import ConvLSTM
from adaptive_s_sgd import AdaptiveSSGD  # not used, but keeps deps consistent
from data_utils import MultiDelayDataset

class ReducerHead(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        z = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.fc(z).squeeze(-1)

def dump_preds(ckpt_path, out_csv, batch_size=16, cpu=False):
    device = torch.device('cpu' if cpu or not torch.cuda.is_available() else 'cuda')
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']
    input_dim = len(cfg['delays'])

    model = ConvLSTM(input_dim=input_dim, hidden_dim=cfg['hidden_dim'], kernel_size=3, num_layers=1).to(device)
    head  = ReducerHead(hidden_dim=cfg['hidden_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    head.load_state_dict(ckpt['head'])
    model.eval(); head.eval()

    ds = MultiDelayDataset(cfg['csv'], target_col='ANNUAL',
                           win_len=cfg['win_len'], delays=tuple(cfg['delays']), split='test')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for X, y, meta in dl:
            X = X.to(device); y = y.to(device)
            seq = [X[:, t] for t in range(X.shape[1])]
            last = model(seq)[-1]
            yhat = head(last)

            # linh hoạt lấy năm dự đoán từ meta
            def as_int_list(obj, n):
                if isinstance(obj, (int, float, str)):
                    try: return [int(obj)] * n
                    except: return [None] * n
                if torch.is_tensor(obj):
                    if obj.ndim == 0: return [int(obj.item())] * n
                    return [int(v) for v in obj.view(-1).tolist()[:n]]
                if isinstance(obj, list):
                    out = []
                    for m in obj:
                        if isinstance(m, dict) and 'year_pred' in m:
                            out.append(int(m['year_pred']))
                        elif torch.is_tensor(m):
                            out.append(int(m.item() if m.ndim==0 else m.view(-1)[0].item()))
                        else:
                            try: out.append(int(m))
                            except: out.append(None)
                    return out
                if isinstance(obj, dict):
                    if 'year_pred' in obj: return as_int_list(obj['year_pred'], n)
                    for v in obj.values(): return as_int_list(v, n)
                    return [None]*n
                return [None]*n

            years = as_int_list(meta, y.shape[0])
            for yi, ypi, yr in zip(y.cpu().tolist(), yhat.cpu().tolist(), years):
                rows.append([yr, yi, ypi])

    pd.DataFrame(rows, columns=['year','y_true_norm','y_pred_norm']).to_csv(out_csv, index=False)
    print(f"[DONE] wrote {out_csv}")

if __name__ == "__main__":
    dump_preds("model_convlstm_adapt_best.pt", "best_preds_test_norm.csv", batch_size=16, cpu=False)
