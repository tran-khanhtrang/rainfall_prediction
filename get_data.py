import os, glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

def find_csv():
    # Tìm đúng file CSV dù nằm ở đâu trong thư mục dự án
    candidates = glob.glob("**/*rainfall*1901*2015*.csv", recursive=True)
    if not candidates:
        raise FileNotFoundError("Không tìm thấy file CSV. Hãy kiểm tra đã giải nén chưa.")
    # Ưu tiên file trong thư mục rainfall_in_india
    candidates.sort(key=lambda p: (0 if "rainfall_in_india" in p.replace("\\","/") else 1, len(p)))
    return candidates[0]

csv_path = find_csv()
print(f"[INFO] Using CSV: {csv_path}")

# Đọc dữ liệu
df = pd.read_csv(csv_path)

# Lọc Tamil Nadu (đảm bảo so sánh không phân biệt hoa thường)
mask = df['SUBDIVISION'].str.upper().str.strip() == 'TAMIL NADU'
tamil = df.loc[mask].copy().reset_index(drop=True)
if tamil.empty:
    raise ValueError("Không tìm thấy hàng cho 'TAMIL NADU' trong CSV.")

# Cột bắt buộc
month_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
assert all(c in tamil.columns for c in month_cols + ['YEAR','ANNUAL']), "CSV thiếu cột tháng hoặc ANNUAL."

# Thêm Year_Index và các quý đúng như bài
tamil['Year_Index'] = tamil['YEAR'] - tamil['YEAR'].min()
tamil['JF']   = tamil[['JAN','FEB']].mean(axis=1)
tamil['MAM']  = tamil[['MAR','APR','MAY']].mean(axis=1)
tamil['JJAS'] = tamil[['JUN','JUL','AUG','SEP']].mean(axis=1)
tamil['OND']  = tamil[['OCT','NOV','DEC']].mean(axis=1)

# Chuẩn hóa min–max về [0,1] cho các trường dùng học
scale_cols = month_cols + ['JF','MAM','JJAS','OND','ANNUAL']
scaler = MinMaxScaler()
tamil[scale_cols] = scaler.fit_transform(tamil[scale_cols])

# Chia 80/20 theo mốc thời gian (giống bài)
cutoff = int(np.floor(0.8 * len(tamil)))
train = tamil.iloc[:cutoff].copy()
test  = tamil.iloc[cutoff:].copy()

# Lưu file
out_train = 'rainfall_tamilnadu_train_1901_1995.csv'
out_test  = 'rainfall_tamilnadu_test_1996_2015.csv'
out_all   = 'rainfall_tamilnadu_1901_2015_ready.csv'

train.to_csv(out_train, index=False)
test.to_csv(out_test, index=False)
tamil.to_csv(out_all, index=False)

print("[DONE] Wrote:", out_train)
print("[DONE] Wrote:", out_test)
print("[DONE] Wrote:", out_all)
