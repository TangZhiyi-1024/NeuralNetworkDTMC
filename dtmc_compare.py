# dtmc_compare.py
# deleted
# 对比两种 DTMC 构建方式：
# 1) 传统频数法（可选分箱）
# 2) 神经网络 + 高斯误差（离散化 CDF 差）

import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

# ======== 配置区 ========
FILE_PATH = r"C:\Users\LENOVO\Desktop\project\sequence.txt"
MODEL_PATH = "sequence_model.h5"
X_SCALER_PATH = "x_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"

VAL_RATIO = 0.10           # 末尾 10% 作为验证（用于对数似然评估）
ALPHA = 0.0                # 拉普拉斯平滑（传统频数 DTMC 用），如需可设 0.5 或 1.0
N_BINS = None              # None=每个唯一值一个状态；或设整数分箱 (如 128)
QUANTILE_BINS = False      # True=分位数分箱；False=等宽分箱
GAUSS_SIGMA_FLOOR = 1e-6   # 高斯法的方差下限
TOPK = 5                   # 打印每行 topK 转移
# ========================


def load_sequence(path):
    seq = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                seq.append(float(s))
    if len(seq) < 3:
        raise ValueError("too short sequence")
    return np.asarray(seq, dtype=np.float64)


def build_dtmc_frequency(sequence, n_bins=None, quantile_bins=False, alpha=0.0):
    """传统频数法：计数相邻对并按行归一。"""
    seq = np.asarray(sequence)

    if n_bins is None:
        states = np.unique(seq)
        S = len(states)
        index = {v: i for i, v in enumerate(states)}
        C = np.zeros((S, S), dtype=np.int64)
        for i in range(len(seq) - 1):
            C[index[seq[i]]], C[index[seq[i+1]]]
            C[index[seq[i]], index[seq[i+1]]] += 1
    else:
        # 分箱
        if quantile_bins:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(seq, qs)
        else:
            mn, mx = float(np.min(seq)), float(np.max(seq))
            edges = np.linspace(mn, mx, n_bins + 1)
        # 宽一点避免边界问题
        edges[0] -= 1e-9; edges[-1] += 1e-9

        states = 0.5 * (edges[:-1] + edges[1:])
        S = len(states)
        b = np.searchsorted(edges, seq, side='right') - 1
        b = np.clip(b, 0, S - 1)
        C = np.zeros((S, S), dtype=np.int64)
        for i in range(len(b) - 1):
            C[b[i], b[i+1]] += 1

    P = C.astype(np.float64)
    if alpha > 0:
        P += alpha

    rowsum = P.sum(axis=1, keepdims=True)
    zero_rows = (rowsum == 0)
    if np.any(zero_rows):
        P[zero_rows[:, 0], :] = 1.0
        rowsum = P.sum(axis=1, keepdims=True)
    P /= rowsum
    return P, states


def gaussian_discrete_row(targets, mu, sigma):
    """离散化高斯：P(k) = Φ(k+0.5) - Φ(k-0.5)（targets 为有序状态值）"""
    left = norm.cdf(targets - 0.5, loc=mu, scale=sigma)
    right = norm.cdf(targets + 0.5, loc=mu, scale=sigma)
    pm = np.maximum(right - left, 0.0)
    s = pm.sum()
    return pm / s if s > 0 else np.ones_like(pm) / len(pm)


def build_dtmc_gaussian(model, x_scaler, y_scaler, sequence, states):
    """神经网络 + 残差高斯近似，输出与 states 对齐的 P。"""
    seq = np.asarray(sequence)

    # 估计全局残差分布（也可做分桶残差，这里用简单版）
    actuals, preds = [], []
    for i in range(len(seq) - 1):
        xi = x_scaler.transform([[seq[i]]])
        yi_hat_s = model.predict(xi, verbose=0)
        yi_hat = y_scaler.inverse_transform(yi_hat_s)[0, 0]
        actuals.append(seq[i + 1]); preds.append(yi_hat)
    residuals = np.asarray(actuals) - np.asarray(preds)
    r_mean = float(np.mean(residuals))
    r_std = float(np.std(residuals, ddof=1))
    sigma = max(r_std, GAUSS_SIGMA_FLOOR)

    # 对所有 states 做一次 NN 预测
    S = len(states)
    xs = x_scaler.transform(np.asarray(states).reshape(-1, 1))
    yhat_s = model.predict(xs, verbose=0).reshape(-1, 1)
    yhat = y_scaler.inverse_transform(yhat_s).reshape(-1)

    mu_vec = yhat + r_mean

    P = np.zeros((S, S), dtype=np.float64)
    for i in range(S):
        P[i] = gaussian_discrete_row(np.asarray(states), mu_vec[i], sigma)

    return P, {'resid_mean': r_mean, 'resid_std': r_std}


def stationary_dist(P, max_iter=10000, tol=1e-12):
    """幂迭代求平稳分布；若不收敛返回 None。"""
    n = P.shape[0]
    pi = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        new_pi = pi @ P
        if np.linalg.norm(new_pi - pi, 1) < tol:
            return new_pi
        pi = new_pi
    return None


def log_likelihood_of_transitions(P, src_idx, dst_idx, eps=1e-12):
    """给定源/目标索引序列，计算对数似然（总和与均值）。"""
    probs = P[src_idx, dst_idx]
    probs = np.clip(probs, eps, 1.0)
    ll = np.log(probs)
    return float(np.sum(ll)), float(np.mean(ll))


def main():
    # 读取数据
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"找不到数据文件：{FILE_PATH}")
    seq = load_sequence(FILE_PATH)
    print(f"总样本点数: {len(seq)}")

    # 切分：末尾 VAL_RATIO 为验证
    X_raw = seq[:-1].reshape(-1, 1)
    y_raw = seq[1:].reshape(-1, 1)
    split = int(len(X_raw) * (1.0 - VAL_RATIO))
    X_tr_raw, y_tr_raw = X_raw[:split], y_raw[:split]
    X_va_raw, y_va_raw = X_raw[split:], y_raw[split:]
    seq_tr = seq[:split+1]   # 训练段的原序列
    seq_va = seq[split:]     # 验证段的原序列

    print(f"训练对数: {len(seq_tr)-1}，验证对数: {len(seq_va)-1}")

    # 1) 传统频数法（在训练段上构建）
    P_freq, states_freq = build_dtmc_frequency(
        seq_tr, n_bins=N_BINS, quantile_bins=QUANTILE_BINS, alpha=ALPHA
    )
    print(f"[Freq] 状态数: {len(states_freq)} | 行和范围: "
          f"{P_freq.sum(axis=1).min():.6f} ~ {P_freq.sum(axis=1).max():.6f}")

    # 2) 高斯/神经网络法（模型需已训练好）
    if not (os.path.exists(MODEL_PATH) and os.path.exists(X_SCALER_PATH) and os.path.exists(Y_SCALER_PATH)):
        raise FileNotFoundError("缺少模型或 scaler 文件。")
    model = load_model(MODEL_PATH, compile=False)  # 只做预测，不需要编译
    x_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    # 为了可比性：高斯法也在同一套 states 上建（这里使用频数法的 states）
    P_gauss, resid_stats = build_dtmc_gaussian(model, x_scaler, y_scaler, seq_tr, states_freq)
    print(f"[Gauss] 行和范围: {P_gauss.sum(axis=1).min():.6f} ~ {P_gauss.sum(axis=1).max():.6f} | "
          f"残差 mean={resid_stats['resid_mean']:.4f}, std={resid_stats['resid_std']:.4f}")

    # 3) Top-K 转移对比（打印前几行示例）
    K = min(TOPK, len(states_freq))
    print("\n=== Top-K 转移对比（前 5 个源状态）===")
    for i in range(min(5, len(states_freq))):
        top_f = np.argsort(-P_freq[i])[:K]
        top_g = np.argsort(-P_gauss[i])[:K]
        print(f"State {states_freq[i]}:")
        print("  Freq :", [(states_freq[j], float(P_freq[i,j])) for j in top_f])
        print("  Gauss:", [(states_freq[j], float(P_gauss[i,j])) for j in top_g])

    # 4) 平稳分布对比
    pi_f = stationary_dist(P_freq)
    pi_g = stationary_dist(P_gauss)
    if pi_f is not None and pi_g is not None:
        l1 = float(np.sum(np.abs(pi_f - pi_g)))
        print(f"\n平稳分布 L1 距离: {l1:.6f}")
        top_f_i = int(np.argmax(pi_f)); top_g_i = int(np.argmax(pi_g))
        print(f"[Freq ] 最常驻状态: {states_freq[top_f_i]} (π={pi_f[top_f_i]:.4f})")
        print(f"[Gauss] 最常驻状态: {states_freq[top_g_i]} (π={pi_g[top_g_i]:.4f})")
    else:
        print("\n平稳分布：至少有一个未收敛（可能链不可约/周期性较强）。")

    # 5) 在验证集上评估转移对数似然
    # 把验证段映射到频数法的 states（若用分箱则天然有映射；若用唯一值，验证里若出现训练没见过的值，会被跳过）
    if N_BINS is None:
        # 唯一值做状态：需要能映射上训练 states，映射不上就跳过
        val_src, val_dst = [], []
        state_to_idx = {v:i for i,v in enumerate(states_freq)}
        for i in range(len(seq_va) - 1):
            s, t = seq_va[i], seq_va[i+1]
            if s in state_to_idx and t in state_to_idx:
                val_src.append(state_to_idx[s]); val_dst.append(state_to_idx[t])
        val_src = np.array(val_src, dtype=np.int64)
        val_dst = np.array(val_dst, dtype=np.int64)
    else:
        # 分箱：复用训练时的边界（states_freq 是区间中心，不包含边界；这里简单近似用最近状态中心来映射）
        # 更严谨可将 edges 保存下来，这里做最近邻近似：
        centers = np.asarray(states_freq)
        def to_bin_idx(x):
            # 最近中心
            return int(np.argmin(np.abs(centers - x)))
        val_src = np.array([to_bin_idx(seq_va[i]) for i in range(len(seq_va)-1)], dtype=np.int64)
        val_dst = np.array([to_bin_idx(seq_va[i+1]) for i in range(len(seq_va)-1)], dtype=np.int64)

    if len(val_src) == 0:
        print("\n[警告] 验证集无法映射到训练的状态空间（可能是唯一值建模且验证出现新值）。")
    else:
        ll_f_sum, ll_f_mean = log_likelihood_of_transitions(P_freq, val_src, val_dst)
        ll_g_sum, ll_g_mean = log_likelihood_of_transitions(P_gauss, val_src, val_dst)
        print("\n=== 验证转移对数似然（越大越好） ===")
        print(f"[Freq ] sum={ll_f_sum:.2f}, mean={ll_f_mean:.6f}")
        print(f"[Gauss] sum={ll_g_sum:.2f}, mean={ll_g_mean:.6f}")

        if ll_g_mean > ll_f_mean:
            print("结论：高斯/NN 模型在验证集上解释力更强。")
        elif ll_g_mean < ll_f_mean:
            print("结论：传统频数模型在验证集上解释力更强。")
        else:
            print("结论：两者相当。")

if __name__ == "__main__":
    main()
