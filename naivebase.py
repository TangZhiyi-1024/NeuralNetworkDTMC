# baseline_eval.py
# 比较“天真基线（y_{t+1} = x_t）”与“当前神经网络模型”的验证集 MAE（原始尺度）

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ======= 配置区 =======
FILE_PATH = r"C:\Users\LENOVO\Desktop\project\sequence.txt"  # 你的序列数据文件
MODEL_PATH = "sequence_model.h5"
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"
VAL_RATIO = 0.10  # 验证集比例（最后10%作为验证）
# =====================

def load_sequence(file_path):
    seq = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                seq.append(int(s))
    if len(seq) < 3:
        raise ValueError("序列太短，无法做验证切分。")
    return seq

def main():
    # 基础检查
    for p in [MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH]:
        if not os.path.exists(p):
            print(f"[错误] 找不到文件：{p}  —— 请确认与本脚本在同一目录或修改路径。")
            sys.exit(1)
    if not os.path.exists(FILE_PATH):
        print(f"[错误] 找不到数据文件：{FILE_PATH}")
        sys.exit(1)

    # 读取数据与模型
    sequence = load_sequence(FILE_PATH)
    print(f"总样本（行）: {len(sequence)}")

    model = load_model(MODEL_PATH, compile=False)
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    # 构造 (x_t -> y_{t+1})
    X_raw = np.array(sequence[:-1], dtype=np.float64).reshape(-1, 1)
    y_raw = np.array(sequence[1:],  dtype=np.float64).reshape(-1, 1)

    # 时序切分：最后 VAL_RATIO 做验证
    split = int(len(X_raw) * (1.0 - VAL_RATIO))
    if split <= 0 or split >= len(X_raw):
        raise ValueError("验证集比例导致切分无效，请调整 VAL_RATIO。")

    X_tr_raw, y_tr_raw = X_raw[:split], y_raw[:split]
    X_va_raw, y_va_raw = X_raw[split:], y_raw[split:]

    print(f"训练样本数: {len(X_tr_raw)}, 验证样本数: {len(X_va_raw)}")

    # ===== A) 天真基线：y_{t+1} = x_t =====
    naive_pred_va = X_va_raw  # 与 y_va_raw 对齐
    naive_mae = float(np.mean(np.abs(naive_pred_va - y_va_raw)))

    # ===== B) 你的模型（原始尺度 MAE）=====
    # 注意：必须用已训练好的 scaler 做 transform / inverse_transform
    X_va = x_scaler.transform(X_va_raw)
    preds_scaled = model.predict(X_va, verbose=0)
    preds = y_scaler.inverse_transform(preds_scaled)
    model_mae = float(np.mean(np.abs(preds - y_va_raw)))

    # 相对误差参考
    data_mean = float(np.mean(y_va_raw))
    data_range = float(np.max(y_va_raw) - np.min(y_va_raw))
    rel_mae_mean = model_mae / (abs(data_mean) if abs(data_mean) > 1e-12 else 1.0)
    rel_mae_range = model_mae / (data_range if data_range > 1e-12 else 1.0)

    print("\n=== 验证集表现（原始尺度）===")
    print(f"[Baseline] Naive   MAE: {naive_mae:.4f}")
    print(f"[Model   ] Current MAE: {model_mae:.4f}")

    print("\n=== 相对误差参考 ===")
    print(f"相对均值 MAE: {rel_mae_mean*100:.2f}%")
    print(f"相对范围 MAE: {rel_mae_range*100:.2f}%")

    # 如果模型没超过基线，给出提示
    if model_mae >= naive_mae:
        print("\n[提示] 当前模型未超过天真基线，可考虑：")
        print("  1) 改用窗口输入（lookback=16/32）；")
        print("  2) 使用 HuberLoss / 调整学习率；")
        print("  3) 对序列做一阶差分建模（预测Δx再还原）；")
        print("  4) 增加模型容量或换用 1D-CNN / LSTM。")

if __name__ == "__main__":
    main()
