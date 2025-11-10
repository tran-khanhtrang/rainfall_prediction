# plot_pred_enhanced.py
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PRED_CSV = Path("preds_test_mm.csv")  # file đã denormalize từ eval_mm.py
METRICS_JSON = Path("model_convlstm_adapt_metrics.json")

def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {PRED_CSV}. Hãy chạy eval_mm.py trước để tạo preds_test_mm.csv."
        )
    df = pd.read_csv(PRED_CSV).dropna()
    if "year" not in df.columns or "y_true_mm" not in df.columns or "y_pred_mm" not in df.columns:
        raise ValueError("preds_test_mm.csv thiếu cột 'year', 'y_true_mm', 'y_pred_mm'")

    df = df.sort_values("year").reset_index(drop=True)

    # Tính residuals và thống kê phụ trợ
    df["residual"] = df["y_pred_mm"] - df["y_true_mm"]
    # Rolling window để ước lượng “độ rộng sai số” theo thời gian
    win = 5
    df["resid_std_roll"] = df["residual"].rolling(win, min_periods=1, center=True).std()
    df["pred_roll"] = df["y_pred_mm"].rolling(3, min_periods=1, center=True).mean()

    # Đọc metrics nếu có (không bắt buộc)
    val_mse, val_prd = None, None
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                m = json.load(f)
                val_mse = m.get("val_mse", None)
                val_prd = m.get("val_prd", None)
        except Exception:
            pass

    # Vẽ
    fig = plt.figure(figsize=(10, 5), dpi=120)
    ax = plt.gca()

    # Đường thực tế & dự báo
    ax.plot(df["year"], df["y_true_mm"], label="Actual (mm)", linewidth=2.0)
    ax.plot(df["year"], df["y_pred_mm"], label="Predicted (mm)", linewidth=2.0, linestyle="--", marker="o", markersize=3.5)

    # Vùng sai số (± rolling σ) quanh dự báo mượt
    upper = df["pred_roll"] + df["resid_std_roll"]
    lower = df["pred_roll"] - df["resid_std_roll"]
    ax.fill_between(df["year"], lower, upper, alpha=0.2, label="±1σ (rolling)")

    # Grid, nhãn, tiêu đề
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Rainfall (mm)")

    title = "Tamil Nadu – Actual vs Predicted (mm)"
    if val_mse is not None and val_prd is not None:
        title += f"\nVal MSE={val_mse:.3f} | Val PRD={val_prd:.2f}%"
    ax.set_title(title)

    ax.legend(loc="best")
    plt.tight_layout()

    # Lưu hình
    png_path = "preds_plot_mm_enhanced.png"
    svg_path = "preds_plot_mm_enhanced.svg"
    plt.savefig(png_path, dpi=150)
    plt.savefig(svg_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

    # Xuất thêm file residuals để bạn kiểm tra chi tiết
    out_csv = "residuals_detail.csv"
    df[["year", "y_true_mm", "y_pred_mm", "residual", "resid_std_roll"]].to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
