#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_future.py (auto-fill history)
- CPU by default
- Loads model 'model_convlstm_adapt.pt'
- Scaler lấy từ --train_csv (ANNUAL mean/std)
- Tự động ghép lịch sử 19012015 từ train_csv nếu future_csv thiếu
- Dự báo khoảng năm chỉ định (mặc định 20162025)
- Ghi file kết quả ở đơn vị mm
"""
import os, argparse
import pandas as pd
import torch
import torch.nn as nn

from convlstm import ConvLSTM

class ReducerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        z = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.fc(z).squeeze(-1)

def build_window_tensor(series_norm, win_len, delays):
    T = win_len
    C = len(delays)
    series = torch.as_tensor(series_norm, dtype=torch.float32)
    end_idx = len(series) - 1
    start_idx = end_idx - (T - 1)
    X = torch.zeros((1, T, C, 1, 1), dtype=torch.float32)
    for t in range(T):
        u = start_idx + t
        for c, d in enumerate(delays):
            idx = u - d
            val = series[max(0, idx)]
            X[0, t, c, 0, 0] = val
    return X

def normalize(vals_mm, mean_mm, std_mm):
    return [(float(x) - mean_mm) / (std_mm + 1e-12) for x in vals_mm]

def denormalize(vals_norm, mean_mm, std_mm):
    return [float(v) * (std_mm + 1e-12) + mean_mm for v in vals_norm]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", "--csv", dest="train_csv",
                   type=str, default="../dataset/rainfall_tamilnadu_1901_2015_ready.csv",
                   help="CSV used to compute scaler (ANNUAL mean/std) & backfill history")
    p.add_argument("--future_csv", type=str, default="rainfall_tamilnadu_future_input.csv",
                   help="Future input CSV (columns: YEAR, ANNUAL; ANNUAL có thể để trống)")
    p.add_argument("--ckpt", type=str, default="model_convlstm_adapt.pt",
                   help="Trained checkpoint file")
    p.add_argument("--win_len", type=int, default=12)
    p.add_argument("--delays", type=int, nargs="+", default=[1,2,3,6,12])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--start_year", type=int, default=2016)
    p.add_argument("--end_year", type=int, default=2025)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--cpu", action="store_true", help="force CPU (default CPU)")
    args = p.parse_args()

    device = torch.device("cpu")

    # 1) scaler từ train csv
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    df_train = pd.read_csv(args.train_csv)
    if "ANNUAL" not in df_train.columns or "YEAR" not in df_train.columns:
        raise ValueError("Train CSV must contain columns 'YEAR' and 'ANNUAL'")
    mean_mm = float(df_train["ANNUAL"].mean())
    std_mm  = float(df_train["ANNUAL"].std())

    # 2) load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg.get("hidden_dim", 128)
    delays     = args.delays if args.delays is not None else cfg.get("delays", [1,2,3])
    win_len    = args.win_len if args.win_len is not None else cfg.get("win_len", 5)

    model = ConvLSTM(input_dim=len(delays), hidden_dim=hidden_dim, kernel_size=3, num_layers=1).to(device)
    head  = ReducerHead(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()

    # 3) đọc future csv (có thể thiếu lịch sử)
    if not os.path.exists(args.future_csv):
        raise FileNotFoundError(f"Future input CSV not found: {args.future_csv}")
    df_future = pd.read_csv(args.future_csv)
    if "YEAR" not in df_future.columns or "ANNUAL" not in df_future.columns:
        raise ValueError("future_csv must contain columns: YEAR, ANNUAL")

    # Chuẩn hoá kiểu & tạo dict cho future
    df_future["YEAR"] = df_future["YEAR"].astype(int)
    future_map = {int(r.YEAR): (None if pd.isna(r.ANNUAL) else float(r.ANNUAL))
                  for _, r in df_future.iterrows()}

    # 4) auto-fill lịch sử từ train (ưu tiên future nếu có giá trị)
    hist_map = {int(r.YEAR): float(r.ANNUAL) for _, r in df_train.iterrows()}
    years_full = sorted(set(list(hist_map.keys()) + list(future_map.keys())))
    # Bắt buộc phải có đủ win_len+max(delays) trước start_year
    min_need = args.start_year - (win_len + max(delays))
    # Đảm bảo có năm từ min_need..end_year (nếu future chưa có, vẫn ok)
    years_full = [y for y in years_full if y <= args.end_year] + \
                 [y for y in range(min_need, args.end_year+1) if y not in years_full]
    years_full = sorted(set(years_full))

    series_mm = []
    for y in years_full:
        if y in future_map and future_map[y] is not None:
            series_mm.append(future_map[y])
        elif y in hist_map:
            series_mm.append(hist_map[y])
        else:
            series_mm.append(float("nan"))

    # Lấy đoạn seed đến end của quá khứ (<= start_year-1) và kiểm tra thiếu
    seed_years = [y for y in years_full if y <= (args.start_year - 1)]
    seed_vals  = [v for y, v in zip(years_full, series_mm) if y <= (args.start_year - 1)]
    if any(pd.isna(v) for v in seed_vals):
        # cố gắng backfill từ train nếu còn thiếu
        fixed = []
        for y in seed_years:
            v = future_map.get(y, None)
            if v is None:
                v = hist_map.get(y, None)
            if v is None:
                raise ValueError(f"Missing history at year {y}; cannot seed model.")
            fixed.append(float(v))
        seed_vals = fixed

    # 5) chuẩn hoá seed và dự báo cuốn chiếu
    series_norm = normalize(seed_vals, mean_mm, std_mm)
    pred_rows = []
    for y in range(args.start_year, args.end_year + 1):
        if len(series_norm) < win_len + max(delays):
            need = win_len + max(delays)
            raise RuntimeError(
                f"Not enough history. Need at least {need} normalized values; got {len(series_norm)}."
            )
        X = build_window_tensor(series_norm, win_len, delays).to(device)
        seq = [X[:, t] for t in range(X.shape[1])]
        with torch.no_grad():
            last = model(seq)[-1]
            yhat_norm = head(last).item()
        yhat_mm = denormalize([yhat_norm], mean_mm, std_mm)[0]
        series_norm.append(yhat_norm)
        pred_rows.append([y, None, float(yhat_mm)])

    out_csv = args.out_csv or f"pred_future_{args.start_year}_{args.end_year}.csv"
    out = pd.DataFrame(pred_rows, columns=["year","y_true_mm","y_pred_mm"])
    out.to_csv(out_csv, index=False)
    print(f"[DONE] wrote {out_csv}")

if __name__ == "__main__":
    main()
