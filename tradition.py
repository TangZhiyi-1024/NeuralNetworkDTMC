import numpy as np

def dtmc_from_sequence(sequence, n_bins=None, quantile_bins=False, alpha=0.0):
    seq = np.asarray(sequence)

    if n_bins is None:
        # 传统：每个唯一值当状态
        states = np.unique(seq)
        S = len(states)
        index = {v:i for i,v in enumerate(states)}
        counts = np.zeros((S, S), dtype=np.int64)

        for i in range(len(seq) - 1):
            src = index[seq[i]]
            dst = index[seq[i+1]]
            counts[src, dst] += 1

    else:
        # 分箱：等宽 or 分位数
        if quantile_bins:
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(seq, qs)
            edges[0] -= 1e-9; edges[-1] += 1e-9
        else:
            mn, mx = float(np.min(seq)), float(np.max(seq))
            edges = np.linspace(mn, mx, n_bins + 1)
            edges[0] -= 1e-9; edges[-1] += 1e-9

        states = 0.5 * (edges[:-1] + edges[1:])   # 用区间中心表示
        S = len(states)

        # 把每个点映到桶索引
        b = np.searchsorted(edges, seq, side='right') - 1
        b = np.clip(b, 0, S - 1)

        counts = np.zeros((S, S), dtype=np.int64)
        for i in range(len(b) - 1):
            counts[b[i], b[i+1]] += 1

    # 归一化（拉普拉斯平滑可选）
    probs = counts.astype(np.float64)
    if alpha > 0:
        probs += alpha

    row_sums = probs.sum(axis=1, keepdims=True)
    # 对零行做保护：均匀分布
    zero_rows = (row_sums == 0).flatten()
    if np.any(zero_rows):
        probs[zero_rows, :] = 1.0
        row_sums[zero_rows, :] = probs[zero_rows, :].sum(axis=1, keepdims=True)

    probs /= row_sums
    return probs, states
