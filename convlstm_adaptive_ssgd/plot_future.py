#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_future.py (enhanced)
Vẽ biểu đồ dự báo mưa kèm Moving Average và Residual Band.
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--save_prefix", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "year" not in df.columns or "y_pred_mm" not in df.columns:
        raise ValueError("CSV must contain columns 'year' and 'y_pred_mm'")

    years = df["year"].astype(int)
    y_pred = df["y_pred_mm"].astype(float)
    y_true = df["y_true_mm"].astype(float) if "y_true_mm" in df.columns else None

    # Moving averages (3-year & 5-year)
    df["pred_ma3"] = y_pred.rolling(window=3, min_periods=1).mean()
    df["pred_ma5"] = y_pred.rolling(window=5, min_periods=1).mean()

    # Nếu có dữ liệu thật, tính residual & 1σ band
    if y_true is not None:
        df["residual"] = y_true - y_pred
        valid_res = df["residual"].dropna()
        if len(valid_res) > 1:
            sigma = valid_res.std()
        else:
            sigma = 0.0
    else:
        sigma = 0.0

    # --- Plot ---
    plt.figure(figsize=(13,6))
    plt.axvspan(years.min(), years.max(), alpha=0.08, color="lightgray", label="Future range")

    # Actual line (if exists)
    if y_true is not None and not df["y_true_mm"].isna().all():
        plt.plot(years, y_true, 'o-', label="Actual (mm)", linewidth=2.2, alpha=0.9)

    # Predicted line
    plt.plot(years, y_pred, 'o-', label="Predicted (mm)", linewidth=2.2, color="tab:blue")

    # Moving averages
    plt.plot(years, df["pred_ma3"], '--', linewidth=1.6, label="Pred MA (3yr)", color="orange")
    plt.plot(years, df["pred_ma5"], '--', linewidth=1.6, label="Pred MA (5yr)", color="green")

    # σ confidence band (nếu có thực tế)
    if y_true is not None and sigma > 0:
        upper = y_pred + sigma
        lower = y_pred - sigma
        plt.fill_between(years, lower, upper, color="blue", alpha=0.1, label=f"1σ  {sigma:.2f} mm")

    # Title & layout
    yr_min, yr_max = years.min(), years.max()
    title = args.title or f"Rainfall Prediction {yr_min}{yr_max}"
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save files
    base = args.save_prefix or os.path.splitext(os.path.basename(args.csv))[0]
    plt.savefig(f"{base}.png", dpi=160)
    plt.savefig(f"{base}.svg")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.svg")

    # Summary
    print(f"Years: {yr_min}{yr_max}")
    print(f"σ (std of residuals): {sigma:.4f} mm")
    print(f"Mean predicted rainfall: {y_pred.mean():.2f} mm")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
