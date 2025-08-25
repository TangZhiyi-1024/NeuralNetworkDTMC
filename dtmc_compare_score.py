# dtmc_eval_both.py
# 同时评传统频数 DTMC 与 NN+高斯 DTMC（同一状态空间），输出 NLL/Brier/Top-k/RMSE/MAE/熵/ECE

import os, numpy as np
from scipy.stats import norm

# ===== 配置 =====
FILE_PATH     = r"C:\Users\LENOVO\Desktop\project\sequence_complete.txt"

# 传统法：从你的 trandition.py 导入（名字就是你之前放的）
from tradition import dtmc_from_sequence   # 需返回 (P_trad, states_trad)

# NN 模型与 scaler
MODEL_PATH    = "sequence_model.h5"
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"

VAL_RATIO     = 0.10        # 末尾10%为验证
N_BINS        = 128         # 传统法分箱个数（建议 64~256）
QUANTILE_BINS = True        # True=分位数分箱（更均衡）
ALPHA_SMOOTH  = 1.0         # 传统法拉普拉斯平滑（每条出边+α）
TOPK_LIST     = [1, 3, 5]
ECE_BINS      = 10
EPS           = 1e-12
BATCH_PRED    = 4096
# ==============

def load_sequence(path):
    xs=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            s=line.strip()
            if s: xs.append(float(s))
    xs=np.asarray(xs,dtype=np.float64)
    if xs.size<3: raise ValueError("序列太短")
    return xs

def load_model_and_scalers():
    from tensorflow.keras.models import load_model
    import joblib
    model = load_model(MODEL_PATH, compile=False)   # 预测用，不反序列化 'mse'
    xsc = joblib.load(X_SCALER_PATH)
    ysc = joblib.load(Y_SCALER_PATH)
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
    # 全局残差
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

    # 逐中心预测
    xs    = xsc.transform(np.asarray(states_centers).reshape(-1,1))
    yhat_s= batched_predict(model, xs, batch=BATCH_PRED).reshape(-1,1)
    yhat  = ysc.inverse_transform(yhat_s).reshape(-1)
    mu_vec= yhat + r_mean

    # CDF 差构建行
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

def eval_metrics(P, states_centers, seq, val_ratio=0.1, topk_list=(1,3,5), ece_bins=10, eps=1e-12):
    X, Y = seq[:-1], seq[1:]
    split = int(len(X)*(1.0-val_ratio))
    Xv, Yv = X[split:], Y[split:]
    if len(Xv)==0: raise ValueError("验证集为空，请调小 VAL_RATIO")

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
        hit = np.any(order[:, :k] == sj[:, None], axis=1)  # 修复广播
        topk_acc[k] = float(np.mean(hit))

    exp_y = (P[si] * states_centers.reshape(1,-1)).sum(axis=1)
    rmse = float(np.sqrt(np.mean((exp_y - Yv)**2)))
    mae  = float(np.mean(np.abs(exp_y - Yv)))

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
    print(f"平均熵(锐度)      : {r['Entropy_mean']:.3f}")
    print(f"ECE(top1)         : {r['ECE_top1']:.3f}")

def main():
    # 读数据 & 切分
    seq = load_sequence(FILE_PATH)
    X, Y = seq[:-1], seq[1:]
    split = int(len(X)*(1.0-VAL_RATIO))
    seq_train = seq[:split+1]

    # 1) 传统频数法：用训练段构建（分箱+平滑）
    P_trad, states_trad = dtmc_from_sequence(
        seq_train, n_bins=N_BINS, quantile_bins=QUANTILE_BINS, alpha=ALPHA_SMOOTH
    )
    P_trad = ensure_row_stochastic(P_trad)
    edges = centers_to_edges(states_trad)  # 供 NN/高斯构建积分用
    print(f"[Trad] 状态数={len(states_trad)} | 行和范围=({P_trad.sum(1).min():.6f}, {P_trad.sum(1).max():.6f})")

    # 2) NN+高斯：在“同一 states_trad”上构建
    model, xsc, ysc = load_model_and_scalers()
    P_gauss = build_gauss_on_states(model, xsc, ysc, seq_train, states_trad, edges)
    P_gauss = ensure_row_stochastic(P_gauss)
    print(f"[Gauss] 行和范围=({P_gauss.sum(1).min():.6f}, {P_gauss.sum(1).max():.6f})")

    # 3) 评估（同一验证集，各自就地映射）
    res_trad  = eval_metrics(P_trad,  states_trad, seq, val_ratio=VAL_RATIO,
                             topk_list=TOPK_LIST, ece_bins=ECE_BINS, eps=EPS)
    res_gauss = eval_metrics(P_gauss, states_trad, seq, val_ratio=VAL_RATIO,
                             topk_list=TOPK_LIST, ece_bins=ECE_BINS, eps=EPS)

    # 4) 打印
    brief("传统频数 DTMC", res_trad)
    brief("NN+高斯 DTMC", res_gauss)

    # 5) 简易结论
    def better(a, b, smaller=True):
        return "Trad" if ((a<b) if smaller else (a>b)) else "Gauss"

    print("\n=== 小结 ===")
    print(f"- 概率更对（NLL↓）：选 **{better(res_trad['NLL_mean'], res_gauss['NLL_mean'], smaller=True)}**")
    print(f"- 点命中更好（Top-1↑）：选 **{'Trad' if res_trad['Topk'][1] > res_gauss['Topk'][1] else 'Gauss'}**")
    print(f"- 期望误差更小（MAE↓）：选 **{better(res_trad['Exp_MAE'], res_gauss['Exp_MAE'], smaller=True)}**")
    print(f"- 校准更好（ECE↓）：选 **{better(res_trad['ECE_top1'], res_gauss['ECE_top1'], smaller=True)}**")

if __name__ == "__main__":
    main()
