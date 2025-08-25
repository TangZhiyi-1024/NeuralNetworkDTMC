# grid_search_simple.py
# 小网格：单点输入的一维回归 x_t -> x_{t+1}
# 评估：验证集“原始尺度 MAE”，并与天真基线比较；保存最佳模型与scaler

import os
import numpy as np
import joblib
import itertools
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, Huber
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.regularizers import l2

# ========== 配置 ==========
FILE_PATH   = r"C:\Users\LENOVO\Desktop\project\sequence_complete.txt"
VAL_RATIO   = 0.10
EPOCHS      = 60           # 不早停，固定轮数
SEED        = 42
USE_RESIDUAL_SKIP = True   # True: y_hat = a*x+b + g(x)（更易超过基线）；False: 纯MLP
L2_WD       = 0.0          # 想更稳可设1e-4
# =========================

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

def build_model(width=64, depth=2, lr=1e-3, loss_name="mse",
                residual=USE_RESIDUAL_SKIP, l2wd=L2_WD):
    inp = Input(shape=(1,), name="x")
    h = inp
    if residual:
        # 主干学残差 g(x)
        for _ in range(depth):
            h = Dense(width, activation='relu',
                      kernel_regularizer=l2(l2wd) if l2wd>0 else None)(h)
        delta = Dense(1, name="delta")(h)
        skip  = Dense(1, use_bias=True, name="skip")(inp)  # a*x+b
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
        # 目标在 [0,1] 归一化尺度，delta 用 0.02 比较稳
        loss_obj = Huber(delta=0.02)
    else:
        raise ValueError("loss_name 必须是 'mse' 或 'huber'")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss_obj,
        metrics=[MeanAbsoluteError()]
    )
    return model

def main():
    # 读数据
    seq = load_sequence(FILE_PATH)
    print(f"总样本点数: {len(seq)}")

    # 构造 (x_t -> x_{t+1})
    X_raw = seq[:-1].reshape(-1, 1)
    y_raw = seq[1:].reshape(-1, 1)

    # 时序切分
    split = int(len(X_raw) * (1.0 - VAL_RATIO))
    X_tr_raw, y_tr_raw = X_raw[:split], y_raw[:split]
    X_va_raw, y_va_raw = X_raw[split:], y_raw[split:]
    print(f"训练样本: {len(X_tr_raw)}, 验证样本: {len(X_va_raw)}")

    # 基线：y_{t+1} = x_t
    naive_mae = float(np.mean(np.abs(X_va_raw - y_va_raw)))
    print(f"[Baseline] Naive MAE (original): {naive_mae:.4f}")

    # 仅用训练集拟合 scaler（防泄漏）
    x_scaler_master = MinMaxScaler()
    y_scaler_master = MinMaxScaler()
    X_tr = x_scaler_master.fit_transform(X_tr_raw)
    y_tr = y_scaler_master.fit_transform(y_tr_raw)
    X_va = x_scaler_master.transform(X_va_raw)
    y_va = y_scaler_master.transform(y_va_raw)

    # 网格
    widths = [32, 64, 128]
    depths = [1, 2]
    losses = ["mse", "huber"]
    lrs    = [1e-3, 3e-4]
    bss    = [32, 64]

    combos = list(itertools.product(widths, depths, losses, lrs, bss))
    print(f"将训练 {len(combos)} 组组合（无早停，每组 {EPOCHS} epochs）...")

    results = []
    best = {"mae_orig": float("inf")}

    # 循环外：初始化“最佳”
    best_mae = float("inf")
    best_config = None


    for i, (w, d, ln, lr, bs) in enumerate(combos, 1):
        # 直接复用同一个 scaler（MinMaxScaler.transform 不会改内部参数）
        x_scaler = x_scaler_master
        y_scaler = y_scaler_master

        model = build_model(width=w, depth=d, lr=lr, loss_name=ln)
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS, batch_size=bs,
            shuffle=False, verbose=0
        )

        # 反归一化后的验证 MAE —— 主指标
        pred_va_scaled = model.predict(X_va, verbose=0)
        pred_va = y_scaler.inverse_transform(pred_va_scaled)
        mae_orig = float(np.mean(np.abs(pred_va - y_va_raw)))

        print(f"[{i:02d}/{len(combos)}] w={w:3d}, d={d}, loss={ln:5s}, lr={lr:.0e}, bs={bs:2d} "
              f"=> Val MAE(orig)={mae_orig:.4f}")

        # 维护最佳（用标量，不用字典）
        if mae_orig < best_mae:
            best_mae = mae_orig
            best_config = {"width": w, "depth": d, "loss": ln, "lr": lr, "batch": bs}
            model.save("best_grid_model.h5")
            joblib.dump(x_scaler_master, "best_x_scaler.pkl")
            joblib.dump(y_scaler_master, "best_y_scaler.pkl")

        # 可选：记录结果列表
        results.append({
            "width": w, "depth": d, "loss": ln, "lr": lr, "batch": bs,
            "val_mae_original": mae_orig,
            "val_mae_scaled": float(history.history["val_mean_absolute_error"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
        })

    # 循环后输出最佳
    print("\n=== 最佳配置 ===")
    print(f"Val MAE(orig)={best_mae:.4f} | "
          f"w={best_config['width']}, d={best_config['depth']}, loss={best_config['loss']}, "
          f"lr={best_config['lr']:.0e}, bs={best_config['batch']}")
    print("已保存：best_grid_model.h5, best_x_scaler.pkl, best_y_scaler.pkl")

    # 排序输出前10
    results_sorted = sorted(results, key=lambda r: r["val_mae_original"])
    print("\n=== 结果排名（按 验证集原始尺度 MAE 从小到大） ===")
    for j, r in enumerate(results_sorted[:10], 1):
        print(f"{j:2d}. w={r['width']:3d}, d={r['depth']}, loss={r['loss']:5s}, "
              f"lr={r['lr']:.0e}, bs={r['batch']:2d} | "
              f"Val MAE(orig)={r['val_mae_original']:.4f}")

    print("\n=== 最佳配置 ===")
    print(f"Val MAE(orig)={best['val_mae_original']:.4f}  | "
          f"w={best['width']}, d={best['depth']}, loss={best['loss']}, "
          f"lr={best['lr']:.0e}, bs={best['batch']}")
    print("已保存：best_grid_model.h5, best_x_scaler.pkl, best_y_scaler.pkl")
    print(f"[Baseline] Naive MAE (original): {naive_mae:.4f}")

if __name__ == "__main__":
    main()
