# 🌧️ Rainfall Prediction using ConvLSTM and Adaptive SSGD

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![VNPT-AI](https://img.shields.io/badge/Organization-VNPT%20AI-blue.svg)](https://vnpt.vn)

> **Author:** TrangTK (Khánh Trang)  
> **Affiliation:** VNPT Software Engineer | MSc in Information Technology  
> **Email:** trangtk.ftu@gmail.com  
> **GitHub:** [github.com/TrangTK](https://github.com/TrangTK)

---

## 🧠 Overview

This project investigates **deep learning techniques for rainfall forecasting** using **Convolutional LSTM (ConvLSTM)** combined with **Adaptive Stochastic Gradient Descent (SSGD)**.  
The goal is to enhance **spatiotemporal rainfall prediction** accuracy based on historical data from the **Kaggle Indian Rainfall Dataset**.

---

## 🎯 Objectives

- Develop a ConvLSTM neural model to forecast rainfall intensity.  
- Experiment with adaptive SSGD optimization for improved training stability.  
- Evaluate predictive performance via MSE, RMSE, MAE, and R².  
- Demonstrate practical AI applications in **climate prediction and sustainability**.

---

## 📂 Project Structure

H:\Neuralbrion\BigData\KaggleDataset
│
├── dataset/ # Preprocessed Kaggle rainfall data
├── convlstm_adaptive_ssgd/ # Deep learning model implementation
├── get_data.py # Script to fetch dataset from Kaggle API
├── venv/ # Virtual environment (ignored in .gitignore)
├── rainfall_in_india/ # Reference raw dataset (if available)
└── README.md # Project documentation (this file)


---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/TrangTK/rainfall_prediction.git
cd rainfall_prediction


2️⃣ Create a Python virtual environment
python -m venv venv
venv\Scripts\activate    # on Windows

3️⃣ Install dependencies
pip install -r requirements.txt


(You can export dependencies later using pip freeze > requirements.txt)

💡 Model Highlights

ConvLSTM: captures both spatial (2D grid) and temporal (sequence) rainfall patterns.

Adaptive SSGD: dynamically adjusts learning rate and momentum to improve training convergence.

Evaluation metrics: MSE, RMSE, MAE, R² for regression accuracy.

📈 Example Results
Metric	Symbol	Value (Example)
Mean Squared Error	MSE	0.0002416
Root Mean Squared Error	RMSE	0.01555
Mean Absolute Error	MAE	0.01210
Coefficient of Determination	R²	0.985

🔍 Research Implications

This study contributes to:

AI-driven climate forecasting and hydrological modeling.

Application of spatiotemporal neural architectures in meteorology.

Integration of AI + Big Data for sustainable development under climate change.

🧩 Future Work

Expand dataset to multiple regions across Asia.

Integrate Transformer-based temporal encoders.

Deploy rainfall prediction API for real-time inference.

🧑‍💻 Author & Contact

TrangTK (Khánh Trang)
Thạc sĩ Công nghệ Thông tin – Kỹ sư Phần mềm VNPT
📧 Email: trangtk.ftu@gmail.com

🌐 GitHub: github.com/TrangTK

📄 License

This project is released for academic and research use under the MIT License.
Please cite or reference if you use parts of this work.


---

## ✅ Sau khi thêm file này:
Trong **VS Code terminal**, chạy:

```bash
git add README.md
git commit -m "Add professional README with project overview and structure"
git push

