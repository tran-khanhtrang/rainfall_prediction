# eval_mm.py
import pandas as pd
import numpy as np
from pathlib import Path

# 1) Đọc dự báo (đang ở thang chuẩn hoá [0,1])
pred_path = Path("model_convlstm_adapt_preds_test_norm.csv")
pred = pd.read_csv(pred_path)

# 2) Lấy min,max từ dữ liệu gốc (CHƯA chuẩn hoá) để khử chuẩn hoá ANNUAL (mm)
#    -> dùng file kaggle gốc đã giải nén ở bước đầu
raw_csv = Path("..") / "rainfall_in_india" / "rainfall in india 1901-2015.csv"
raw = pd.read_csv(raw_csv)
tn = raw[raw['SUBDIVISION'].str.upper().str.strip() == 'TAMIL NADU'].copy()
tn = tn.sort_values('YEAR').reset_index(drop=True)

y_mm = tn['ANNUAL'].astype(float).values
y_min, y_max = y_mm.min(), y_mm.max()

def inv_minmax(x): 
    return x * (y_max - y_min) + y_min

# 3) Khử chuẩn hoá
pred['y_true_mm'] = pred['y_true_norm'].apply(inv_minmax)
pred['y_pred_mm'] = pred['y_pred_norm'].apply(inv_minmax)

# 4) Metric trên mm
mse_mm = float(np.mean((pred['y_true_mm'] - pred['y_pred_mm'])**2))
prd_mm = float(100 * np.sqrt(np.sum((pred['y_true_mm'] - pred['y_pred_mm'])**2) /
                             (np.sum(pred['y_true_mm']**2) + 1e-12)))

print(f"MSE (mm^2): {mse_mm:.3f}")
print(f"PRD on mm (%): {prd_mm:.2f}")

# 5) Lưu file kiểm chứng
pred.to_csv("preds_test_mm.csv", index=False)
print("Wrote: preds_test_mm.csv")
