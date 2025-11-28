# Simultaneously evaluate traditional frequency DTMC and NN+Gaussian DTMC (same state space)
# outputting NLL/Brier/Top-k/RMSE/MAE/Entropy/ECE

import os, numpy as np
from scipy.stats import norm
from tensorflow.keras.models import load_model
import joblib
from tradition import dtmc_from_sequence   # return (P_trad, states_trad)
FILE_PATH     = r"C:\Users\LENOVO\Desktop\project\sequence_complete.txt"

# NN and scaler
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"

VAL_RATIO     = 0.10        # validation: 10%
N_BINS        = 128         # Number of bins for traditional methods
QUANTILE_BINS = True
ALPHA_SMOOTH  = 1.0
TOPK_LIST     = [1, 3, 5]
ECE_BINS      = 10
EPS           = 1e-12
BATCH_PRED    = 4096
TOL = 200.0   # <=200 视为识别正确



def load_sequence(path):
    xs=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            s=line.strip()
            if s: xs.append(float(s))
    xs=np.asarray(xs,dtype=np.float64)
    if xs.size<3: raise ValueError("sequence too short")
    return xs


def load_model_and_scalers(model_path, x_scaler_path, y_scaler_path):
    model = load_model(model_path, compile=False)
    xsc = joblib.load(x_scaler_path)
    ysc = joblib.load(y_scaler_path)
    return model, xsc, ysc


def centers_to_edges(centers):
    c = np.asarray(centers, dtype=np.float64)
    if len(c)==1:
        edges = np.array([c[0]-0.5, c[0]+0.5], dtype=np.float64)
    else:
        dif = np.diff(c)
        left  = c[0]  - dif[0]/2.0
        right = c[-1] + dif[-1]/2.0
        mids = c[:-1] + dif/2.0
        edges = np.concatenate([[left], mids, [right]])
    edges[0] -= 1e-9; edges[-1] += 1e-9
    return edges

def batched_predict(model, X, batch=BATCH_PRED):
    outs=[]
    for i in range(0, len(X), batch):
        outs.append(model.predict(X[i:i+batch], verbose=0))
    return np.vstack(outs)

def build_gauss_on_states(model, xsc, ysc, seq_train, states_centers, edges, sigma_floor=1e-6):
    # Global residual
    actuals, preds = [], []
    for t in range(len(seq_train)-1):
        xi = xsc.transform([[seq_train[t]]])
        yhat_s = model.predict(xi, verbose=0)
        yhat   = ysc.inverse_transform(yhat_s)[0,0]
        actuals.append(seq_train[t+1]); preds.append(yhat)
    resid = np.asarray(actuals) - np.asarray(preds)
    r_mean = float(np.mean(resid))
    r_std  = float(np.std(resid, ddof=1))
    sigma  = max(r_std, sigma_floor)

    # Center-by-center prediction
    xs    = xsc.transform(np.asarray(states_centers).reshape(-1,1))
    yhat_s= batched_predict(model, xs, batch=BATCH_PRED).reshape(-1,1)
    yhat  = ysc.inverse_transform(yhat_s).reshape(-1)
    mu_vec= yhat + r_mean

    # CDF
    S = len(states_centers)
    P = np.zeros((S,S), dtype=np.float64)
    eL, eR = edges[:-1], edges[1:]
    for i in range(S):
        pm = np.maximum(norm.cdf(eR, loc=mu_vec[i], scale=sigma) -
                        norm.cdf(eL, loc=mu_vec[i], scale=sigma), 0.0)
        s = pm.sum()
        P[i] = pm/s if s>0 else np.ones(S)/S
    return P

def ensure_row_stochastic(P):
    P = np.asarray(P, dtype=np.float64)
    rowsum = P.sum(axis=1, keepdims=True)
    zero = (rowsum<=0) | ~np.isfinite(rowsum)
    if np.any(zero):
        P[zero[:,0],:] = 1.0
        rowsum = P.sum(axis=1, keepdims=True)
    return P/rowsum

def map_to_idx_by_nearest(states, values):
    states = np.asarray(states); values = np.asarray(values)
    return np.abs(values.reshape(-1,1)-states.reshape(1,-1)).argmin(axis=1)

def eval_metrics(P, states_centers, seq, val_ratio=0.1, topk_list=(1,3,5),
                 ece_bins=10, eps=1e-12, tol=100.0):
    X, Y = seq[:-1], seq[1:]
    split = int(len(X)*(1.0-val_ratio))
    Xv, Yv = X[split:], Y[split:]
    if len(Xv)==0: raise ValueError("The validation set is empty")

    si = map_to_idx_by_nearest(states_centers, Xv)
    sj = map_to_idx_by_nearest(states_centers, Yv)

    p_true = np.clip(P[si, sj], eps, 1.0)
    nll_mean = -float(np.mean(np.log(p_true)))

    row_sqsum = (P[si]**2).sum(axis=1)
    brier_mean = float(np.mean(row_sqsum + (1.0 - 2.0*p_true)))

    order = np.argsort(-P[si], axis=1)
    topk_acc = {}
    for k in topk_list:
        k = min(k, P.shape[1])
        hit = np.any(order[:, :k] == sj[:, None], axis=1)
        topk_acc[k] = float(np.mean(hit))

    # === 数值预测(期望) ===
    exp_y = (P[si] * states_centers.reshape(1,-1)).sum(axis=1)
    rmse = float(np.sqrt(np.mean((exp_y - Yv)**2)))
    mae  = float(np.mean(np.abs(exp_y - Yv)))

    # === 新增：容差准确率 ===
    tol_acc = float(np.mean(np.abs(exp_y - Yv) <= tol))

    probs = np.clip(P[si], eps, 1.0)
    entropy = float(np.mean(-(probs*np.log(probs)).sum(axis=1)))

    # Top1 ECE
    p_top1 = P[si, order[:,0]]
    hit_top1 = (order[:,0] == sj).astype(np.float64)
    bins = np.linspace(0,1,ece_bins+1)
    idx  = np.clip(np.digitize(p_top1, bins)-1, 0, ece_bins-1)
    ece=0.0
    for b in range(ece_bins):
        m = (idx==b)
        if not np.any(m): continue
        conf_b = float(np.mean(p_top1[m]))
        acc_b  = float(np.mean(hit_top1[m]))
        ece += abs(acc_b - conf_b) * (np.sum(m)/len(hit_top1))

    S = P.shape[1]
    nll_uniform = -float(np.log(1.0/max(S,1)))
    return {
        "N_valid": len(Xv),
        "NLL_mean": nll_mean,
        "NLL_uniform": nll_uniform,
        "Brier_mean": brier_mean,
        "Topk": topk_acc,
        "Exp_RMSE": rmse,
        "Exp_MAE": mae,
        "TolAcc": tol_acc,              # <-- 新增
        "Entropy_mean": entropy,
        "ECE_top1": ece,
    }


def brief(name, r):
    print(f"\n== {name} ==")
    print(f"N_valid            : {r['N_valid']}")
    print(f"NLL (mean)        : {r['NLL_mean']:.6f}   | Uniform: {r['NLL_uniform']:.6f}")
    print(f"Brier (mean)      : {r['Brier_mean']:.6f}")
    print("Top-k 命中        : " + ", ".join([f"@{k}={r['Topk'][k]:.3f}" for k in sorted(r['Topk'])]))
    print(f"期望预测 RMSE/MAE : {r['Exp_RMSE']:.3f} / {r['Exp_MAE']:.3f}")
    print(f"容差准确率(|err|≤{TOL:g}) : {r['TolAcc']:.3f}")   # <-- 新增
    print(f"平均熵(锐度)      : {r['Entropy_mean']:.3f}")
    print(f"ECE(top1)         : {r['ECE_top1']:.3f}")

def main():
    # Read data & split
    seq = load_sequence(FILE_PATH)
    X, Y = seq[:-1], seq[1:]
    split = int(len(X)*(1.0-VAL_RATIO))
    seq_train = seq[:split+1]

    # tradition DTMC (只算一次)
    P_trad, states_trad = dtmc_from_sequence(
        seq_train, n_bins=N_BINS, quantile_bins=QUANTILE_BINS, alpha=ALPHA_SMOOTH
    )
    P_trad = ensure_row_stochastic(P_trad)
    edges = centers_to_edges(states_trad)
    print(f"[Trad] 状态数={len(states_trad)} | 行和范围=({P_trad.sum(1).min():.6f}, {P_trad.sum(1).max():.6f})")

    # 多个模型
    MODEL_SPECS = [
        {
            "name": "NN_base",
            "model": "sequence_model.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
        {
            "name": "NN_v1",
            "model": "sequence_model_1.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
        {
            "name": "NN_v2",
            "model": "sequence_model_2.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
        {
            "name": "NN_v3",
            "model": "sequence_model_3.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
        {
            "name": "NN_v4",
            "model": "sequence_model_4.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
        {
            "name": "NN_v5",
            "model": "sequence_model_5.h5",
            "xsc": "x_scaler.pkl",
            "ysc": "y_scaler.pkl",
        },
    ]

    # evaluate Trad 作为 baseline
    res_trad = eval_metrics(P_trad, states_trad, seq, val_ratio=VAL_RATIO,
                            topk_list=TOPK_LIST, ece_bins=ECE_BINS, eps=EPS, tol=TOL)

    results = [("Traditional DTMC", res_trad)]

    # ==== 对每个 NN 模型构建 Gauss DTMC 并评估 ====
    for spec in MODEL_SPECS:
        model, xsc, ysc = load_model_and_scalers(spec["model"], spec["xsc"], spec["ysc"])
        P_gauss = build_gauss_on_states(model, xsc, ysc, seq_train, states_trad, edges)
        P_gauss = ensure_row_stochastic(P_gauss)
        print(f"[{spec['name']}] 行和范围=({P_gauss.sum(1).min():.6f}, {P_gauss.sum(1).max():.6f})")

        res = eval_metrics(P_gauss, states_trad, seq, val_ratio=VAL_RATIO,
                           topk_list=TOPK_LIST, ece_bins=ECE_BINS, eps=EPS, tol=TOL)
        results.append((spec["name"] + "+Gauss DTMC", res))

    for name, r in results:
        brief(name, r)

    # ==== 汇总对比（按 TolAcc / MAE 排序）====
    print("\n=== Summary (sorted by TolAcc desc) ===")
    results_sorted = sorted(results, key=lambda x: x[1]["TolAcc"], reverse=True)
    for name, r in results_sorted:
        print(f"{name:20s} | TolAcc={r['TolAcc']:.4f} | MAE={r['Exp_MAE']:.1f} | Top1={r['Topk'][1]:.3f} | NLL={r['NLL_mean']:.3f}")


if __name__ == "__main__":
    main()
