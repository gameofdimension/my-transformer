import torch

from alternatives.tools import timer_func


def recursive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: float):
    """
    Args:
        q (torch.Tensor): (n, d)
        k (torch.Tensor): (n, d)
        v (torch.Tensor): (n, d)
    """
    def accumulate(query, last):
        if last == 0:
            return k[last].dot(query)*v[last]
        return gamma * accumulate(query, last - 1) + k[last].dot(query)*v[last]

    out = torch.zeros_like(v)
    seq_len = q.shape[0]
    for i in range(seq_len):
        out[i] = accumulate(q[i], i)
    return out


def iterative(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: float):
    out = torch.zeros_like(v)
    seq_len = q.shape[0]
    for i in range(seq_len):
        query = q[i]
        for j in range(i, -1, -1):
            out[i] += k[j].dot(query)*v[j]*gamma**(i-j)
    return out


def make_gamma_matrix(gamma: float, seq_len: int):
    gamma_matrix = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(i, -1, -1):
            gamma_matrix[i, j] = gamma**(i-j)
    return gamma_matrix


def parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: float):
    """
    both q and k are already rotated
    """
    return ((q@k.T)*mask)@v


def main():
    """
    展示可以用 parallel 方式实现 retnet 的 attention
    在 cpu 上运行，我们就可以看到数百倍的加速

    Function 'recursive' executed in 2.1209s
    Function 'iterative' executed in 2.5344s
    Function 'parallel' executed in 0.0035s
    tensor(0.0002)
    tensor(0.0001)
    tensor(0.0001)
    """
    seq_len = 512
    dim = 256
    q = torch.randn(seq_len, dim)
    k = torch.randn(seq_len, dim)
    v = torch.randn(seq_len, dim)
    gamma = 0.9
    mask = make_gamma_matrix(gamma, seq_len)
    out1 = timer_func(recursive)(q, k, v, gamma)
    out2 = timer_func(iterative)(q, k, v, gamma)
    out3 = timer_func(parallel)(q, k, v, mask)
    print((out1-out2).abs().max())
    print((out1-out3).abs().max())
    print((out2-out3).abs().max())


if __name__ == "__main__":
    main()
