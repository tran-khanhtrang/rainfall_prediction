# plot_pred.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("preds_test_mm.csv").dropna()
df = df.sort_values("year")

plt.figure()
plt.plot(df["year"], df["y_true_mm"], label="Actual (mm)")
plt.plot(df["year"], df["y_pred_mm"], label="Predicted (mm)")
plt.xlabel("Year"); plt.ylabel("Annual Rainfall (mm)")
plt.title("Tamil Nadu – Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("preds_plot_mm.png", dpi=150)
print("Saved: preds_plot_mm.png")
