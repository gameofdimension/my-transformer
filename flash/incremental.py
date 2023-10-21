import numpy as np


def incremental_softmax_weighted_sum(score: np.ndarray, value: np.ndarray, block_size: int):
    assert score.ndim == 1
    assert score.shape[0] == value.shape[0]

    def compute_one_block(s: np.ndarray, val: np.ndarray):
        assert 0 < s.shape[0] <= block_size
        assert s.shape[0] == val.shape[0]

        max_expo = np.max(s)
        s = s - max_expo
        exp_val = np.exp(s)
        denominator = np.sum(exp_val)
        return max_expo, exp_val @ val, denominator

    def merge_two_block(
            numerator1: np.ndarray, denominator1: np.ndarray, expo1: np.ndarray,
            numerator2: np.ndarray, denominator2: np.ndarray, expo2: np.ndarray):
        max_expo = np.maximum(expo1, expo2)
        total = denominator1 * np.exp(expo1 - max_expo) + denominator2 * np.exp(expo2 - max_expo)

        numerator1 = np.exp(expo1 - max_expo) * numerator1
        numerator2 = np.exp(expo2 - max_expo) * numerator2
        return max_expo, numerator1 + numerator2, total

    i = 0
    max_e, num, denom = np.array(-float('inf')), 0, 0

    while True:
        if i + block_size >= score.shape[0]:
            new_max_e, new_num, new_denom = compute_one_block(score[i:], value[i:, :])
            _, num, denom = merge_two_block(num, denom, max_e, new_num, new_denom, new_max_e)
            return num / denom

        new_max_e, new_num, new_denom = compute_one_block(score[i:i + block_size], value[i:i + block_size, :])
        max_e, num, denom = merge_two_block(num, denom, max_e, new_num, new_denom, new_max_e)
        i += block_size


def incremental_softmax_weighted_sum_2d(score: np.ndarray, value: np.ndarray, block_size: int):
    """

    :param score: [r, seq length]
    :param value: [seq length, d]
    :param block_size:
    :return:
    """
    assert len(score.size()) == len(value.size()) == 2
    assert score.size(1) == value.size(0)

    def compute_one_block(vec: np.ndarray, val: np.ndarray):
        assert 0 < vec.size(1) <= block_size
        assert vec.size(1) == val.size(0)
        maxv = vec.max(axis=1, keepdims=True)

        vec = vec - maxv
        exp_val = np.exp(vec)
        total = np.sum(exp_val, axis=1, keepdims=True)
        return maxv, exp_val @ val, total

    def merge_two_block(
            v1: np.ndarray, l1: np.ndarray, m1: np.ndarray,
            v2: np.ndarray, l2: np.ndarray, m2: np.ndarray):
        maxv = np.maximum(m1, m2)
        total = l1 * np.exp(m1 - maxv) + l2 * np.exp(m2 - maxv)

        v1 = np.exp(m1 - maxv) * v1
        v2 = np.exp(m2 - maxv) * v2
        return maxv, v1 + v2, total

    def iterative_merge():
        m = np.full((score.size(0), 1), -float('inf'))
        e = np.zeros((score.size(0), value.size(1)))
        t = np.zeros((score.size(0), 1))

        i = 0
        while True:
            if i + block_size >= score.size(1):
                nm, ne, nt = compute_one_block(score[:, i:], value[i:, :])
                _, e, t = merge_two_block(e, t, m, ne, nt, nm)
                return e / t

            nm, ne, nt = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
            m, e, t = merge_two_block(e, t, m, ne, nt, nm)
            i += block_size

    def merge_all_at_once():
        lst = []
        maxm = np.full((score.size(0), 1), -float('inf'))
        for i in range(0, score.size(1), block_size):
            m, n, d = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
            lst.append((m, n, d))
            assert maxm.size() == m.size()
            maxm = np.maximum(maxm, m)

        ns = np.zeros((score.size(0), value.size(1)))
        ds = np.zeros((score.size(0), 1))
        for m, n, d in lst:
            ns += np.exp(m - maxm) * n
            ds += np.exp(m - maxm) * d

        return ns / ds

    return iterative_merge(), merge_all_at_once()
