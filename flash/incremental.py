import torch


def incremental_softmax_weighted_sum(score: torch.Tensor, value: torch.Tensor, block_size: int):
    assert len(score.size()) == 1
    assert score.size(0) == value.size(0)

    def compute_one_block(vec: torch.Tensor, val: torch.Tensor):
        assert 0 < vec.size(0) <= block_size
        assert vec.size(0) == val.size(0)

        maxv = torch.max(vec)
        vec = vec - maxv
        exp_val = torch.exp(vec)
        total = torch.sum(exp_val)
        return maxv, exp_val @ val, total

    def merge_two_block(
            v1: torch.Tensor, l1: torch.Tensor, m1: torch.Tensor,
            v2: torch.Tensor, l2: torch.Tensor, m2: torch.Tensor):
        maxv = torch.max(m1, m2)
        total = l1 * torch.exp(m1 - maxv) + l2 * torch.exp(m2 - maxv)

        v1 = torch.exp(m1 - maxv) * v1
        v2 = torch.exp(m2 - maxv) * v2
        return maxv, v1 + v2, total

    i = 0
    m, e, t = torch.tensor(-float('inf')), 0, 0

    while True:
        if i + block_size >= score.size(0):
            nm, ne, nt = compute_one_block(score[i:], value[i:, :])
            _, e, t = merge_two_block(e, t, m, ne, nt, nm)
            return e / t

        nm, ne, nt = compute_one_block(score[i:i + block_size], value[i:i + block_size, :])
        m, e, t = merge_two_block(e, t, m, ne, nt, nm)
        i += block_size


def incremental_softmax_weighted_sum_2d(score: torch.Tensor, value: torch.Tensor, block_size: int):
    """

    :param score: [r, seq length]
    :param value: [seq length, d]
    :param block_size:
    :return:
    """
    assert len(score.size()) == len(value.size()) == 2
    assert score.size(1) == value.size(0)

    def compute_one_block(vec: torch.Tensor, val: torch.Tensor):
        assert 0 < vec.size(1) <= block_size
        assert vec.size(1) == val.size(0)
        maxv = torch.max(vec, dim=1, keepdim=True).values

        vec = vec - maxv
        exp_val = torch.exp(vec)
        total = torch.sum(exp_val, dim=1, keepdim=True)
        return maxv, exp_val @ val, total

    def merge_two_block(
            v1: torch.Tensor, l1: torch.Tensor, m1: torch.Tensor,
            v2: torch.Tensor, l2: torch.Tensor, m2: torch.Tensor):
        maxv = torch.maximum(m1, m2)
        total = l1 * torch.exp(m1 - maxv) + l2 * torch.exp(m2 - maxv)

        v1 = torch.exp(m1 - maxv) * v1
        v2 = torch.exp(m2 - maxv) * v2
        return maxv, v1 + v2, total

    def iterative_merge():
        m = torch.full((score.size(0), 1), -float('inf'))
        e = torch.zeros((score.size(0), value.size(1)))
        t = torch.zeros((score.size(0), 1))

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
        maxm = torch.full((score.size(0), 1), -float('inf'))
        for i in range(0, score.size(1), block_size):
            m, n, d = compute_one_block(score[:, i:i + block_size], value[i:i + block_size, :])
            lst.append((m, n, d))
            assert maxm.size() == m.size()
            maxm = torch.maximum(maxm, m)

        ns = torch.zeros((score.size(0), value.size(1)))
        ds = torch.zeros((score.size(0), 1))
        for m, n, d in lst:
            ns += torch.exp(m - maxm) * n
            ds += torch.exp(m - maxm) * d

        return ns / ds

    return iterative_merge(), merge_all_at_once()
