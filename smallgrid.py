# grid_search_fixed_lr.py
# 固定学习率(Adam lr=3e-4)的小网格：单点输入 x_t -> x_{t+1}
# 评估：验证集原始尺度 MAE；保存最佳模型与 scaler

import os
import numpy as np
import itertools
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.regularizers import l2

# ========= 配置 =========
FILE_PATH   = r"C:\Users\LENOVO\Desktop\project\sequence.txt"
VAL_RATIO   = 0.10
EPOCHS      = 100              # 固定轮数，不早停
SEED        = 42
LR_FIXED    = 3e-4             # <<< 固定学习率（改这里）
USE_RESIDUAL_SKIP = True       # 线性跳连：ŷ = a*x + b + g(x)
L2_WD       = 0.0              # 如需更稳可设 1e-4
# =======================

tf.keras.utils.set_random_seed(SEED)
tf.get_logger().setLevel("ERROR")

def load_sequence(path):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                xs.append(float(s))
    if len(xs) < 3:
        raise ValueError("序列太短")
    return np.asarray(xs, dtype=np.float64)

def build_model(width=64, depth=2, loss_name="mse",
                residual=USE_RESIDUAL_SKIP, l2wd=L2_WD, lr=LR_FIXED):
    inp = Input(shape=(1,), name="x")
    h = inp
    if residual:
        # 主干学非线性残差 g(x)
        for _ in range(depth):
            h = Dense(width, activation='relu',
                      kernel_regularizer=l2(l2wd) if l2wd>0 else None)(h)
        delta = Dense(1, name="delta")(h)
        skip  = Dense(1, use_bias=True, name="skip")(inp)  # a*x + b
        out   = Add(name="y_hat")([skip, delta])
    else:
        for _ in range(depth):
            h = Dense(width, activation='relu',
                      kernel_regularizer=l2(l2wd) if l2wd>0 else None)(h)
        out = Dense(1, name="y_hat")(h)

    model = Model(inp, out)

    if loss_name == "mse":
        loss_obj = MeanSquaredError()
    elif loss_name == "huber":
        # 在 [0,1] 归一化尺度上，delta=0.02 较稳
        loss_obj = Huber(delta=0.02)
    else:
        raise ValueError("loss_name 必须是 'mse' 或 'huber'")

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=loss_obj,
                  metrics=[MeanAbsoluteError()])
    return model

def main():
    # 读数据
    seq = load_sequence(FILE_PATH)
    print(f"总样本点数: {len(seq)}")

    # 构造 x_t -> x_{t+1}
    X_raw = seq[:-1].reshape(-1, 1)
    y_raw = seq[1:].reshape(-1, 1)

    # 时序切分：最后 10% 作为验证
    split = int(len(X_raw) * (1.0 - VAL_RATIO))
    X_tr_raw, y_tr_raw = X_raw[:split], y_raw[:split]
    X_va_raw, y_va_raw = X_raw[split:], y_raw[split:]
    print(f"训练样本: {len(X_tr_raw)}, 验证样本: {len(X_va_raw)}")

    # 天真基线：y_{t+1} = x_t
    naive_mae = float(np.mean(np.abs(X_va_raw - y_va_raw)))
    print(f"[Baseline] Naive MAE (original): {naive_mae:.4f}")

    # 仅用训练集拟合 scaler（防泄漏）
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr_raw)
    y_tr = y_scaler.fit_transform(y_tr_raw)
    X_va = x_scaler.transform(X_va_raw)
    y_va = y_scaler.transform(y_va_raw)

    # ---------- 小网格（固定学习率） ----------
    widths = [32, 64, 128]
    depths = [1, 2, 3]          # 包含 d>=2
    losses = ["mse", "huber"]
    batch_sizes = [32, 64]

    combos = list(itertools.product(widths, depths, losses, batch_sizes))
    print(f"将训练 {len(combos)} 组组合（Adam lr 固定为 {LR_FIXED}，每组 {EPOCHS} epochs）...")

    best_mae = float("inf")
    best_cfg = None
    results = []

    for i, (w, d, ln, bs) in enumerate(combos, 1):
        model = build_model(width=w, depth=d, loss_name=ln,
                            residual=USE_RESIDUAL_SKIP, l2wd=L2_WD, lr=LR_FIXED)
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS,
            batch_size=bs,
            shuffle=False,
            verbose=0
        )

        # 验证集原始尺度 MAE（主指标）
        pred_va_scaled = model.predict(X_va, verbose=0)
        pred_va = y_scaler.inverse_transform(pred_va_scaled)
        mae_orig = float(np.mean(np.abs(pred_va - y_va_raw)))

        print(f"[{i:02d}/{len(combos)}] w={w:3d}, d={d}, loss={ln:5s}, bs={bs:2d} "
              f"=> Val MAE(orig)={mae_orig:.4f}")

        # 维护最佳
        if mae_orig < best_mae:
            best_mae = mae_orig
            best_cfg = {"width": w, "depth": d, "loss": ln, "batch": bs}
            model.save("best_grid_model_fixedlr.h5")
            joblib.dump(x_scaler, "best_x_scaler.pkl")
            joblib.dump(y_scaler, "best_y_scaler.pkl")

        # 记录
        results.append({
            "width": w, "depth": d, "loss": ln, "batch": bs,
            "val_mae_original": mae_orig,
            "val_mae_scaled": float(hist.history["val_mean_absolute_error"][-1]),
            "val_loss": float(hist.history["val_loss"][-1]),
        })

    # 排名输出前 10
    results_sorted = sorted(results, key=lambda r: r["val_mae_original"])
    print("\n=== 结果排名（按 验证集原始尺度 MAE 从小到大） ===")
    for j, r in enumerate(results_sorted[:10], 1):
        print(f"{j:2d}. w={r['width']:3d}, d={r['depth']}, loss={r['loss']:5s}, "
              f"bs={r['batch']:2d} | Val MAE(orig)={r['val_mae_original']:.4f}")

    # 最佳汇总
    print("\n=== 最佳配置（固定 lr） ===")
    print(f"Val MAE(orig)={best_mae:.4f} | "
          f"w={best_cfg['width']}, d={best_cfg['depth']}, loss={best_cfg['loss']}, bs={best_cfg['batch']}")
    print("已保存：best_grid_model_fixedlr.h5, best_x_scaler.pkl, best_y_scaler.pkl")
    print(f"[Baseline] Naive MAE (original): {naive_mae:.4f}")

if __name__ == "__main__":
    main()
