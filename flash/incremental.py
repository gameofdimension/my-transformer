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
    assert score.ndim == value.ndim == 2
    assert score.shape[1] == value.shape[0]

    def compute_one_block(s: np.ndarray, val: np.ndarray):
        assert 0 < s.shape[1] <= block_size
        assert s.shape[1] == val.shape[0]
        max_expo = s.max(axis=1, keepdims=True)

        s = s - max_expo
        exp_val = np.exp(s)
        denominator = np.sum(exp_val, axis=1, keepdims=True)
        return max_expo, exp_val @ val, denominator

    def merge_two_block(
            numerator1: np.ndarray, denominator1: np.ndarray, expo1: np.ndarray,
            numerator2: np.ndarray, denominator2: np.ndarray, expo2: np.ndarray):
        max_expo = np.maximum(expo1, expo2)
        total = denominator1 * np.exp(expo1 - max_expo) + denominator2 * np.exp(expo2 - max_expo)

        numerator1 = np.exp(expo1 - max_expo) * numerator1
        numerator2 = np.exp(expo2 - max_expo) * numerator2
        return max_expo, numerator1 + numerator2, total

    def iterative_merge():
        max_e = np.full((score.shape[0], 1), -float('inf'))
        num = np.zeros((score.shape[0], value.shape[1]))
        denom = np.zeros((score.shape[0], 1))

        i = 0
        while True:
            if i + block_size >= score.shape[1]:
                new_max_e, new_num, new_denom = compute_one_block(score[:, i:], value[i:, :])
                _, num, denom = merge_two_block(num, denom, max_e, new_num, new_denom, new_max_e)
                return num / denom

            new_max_e, new_num, new_denom = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
            max_e, num, denom = merge_two_block(num, denom, max_e, new_num, new_denom, new_max_e)
            i += block_size

    def merge_all_at_once():
        lst = []
        max_expo = np.full((score.shape[0], 1), -float('inf'))
        for i in range(0, score.shape[1], block_size):
            m, n, d = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
            lst.append((m, n, d))
            assert max_expo.shape == m.shape
            max_expo = np.maximum(max_expo, m)

        ns = np.zeros((score.shape[0], value.shape[1]))
        ds = np.zeros((score.shape[0], 1))
        for m, n, d in lst:
            ns += np.exp(m - max_expo) * n
            ds += np.exp(m - max_expo) * d

        return ns / ds

    return iterative_merge(), merge_all_at_once()
