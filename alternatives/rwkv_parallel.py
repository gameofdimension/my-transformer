import torch

from alternatives.tools import timer_func


def iterative(
        key: torch.Tensor, value: torch.Tensor,
        u: torch.Tensor, w: torch.Tensor,
        device, dtype):
    '''
    adapt from model/rwkv.py
    '''
    seq_length, hidden_size = key.size()
    assert u.dim() == w.dim() == 1
    assert u.size(-1) == hidden_size
    assert key.size(-1) == value.size(-1)

    lst = []
    a = torch.zeros(hidden_size, device=device, dtype=dtype)
    b = torch.zeros_like(a)
    exponent = torch.full_like(a, -float('inf'))
    for t in range(seq_length):
        kt = key[t]

        # compute wkv
        max_exponent = torch.max(exponent, u + kt)
        wt = torch.exp(u + kt - max_exponent)
        vt = value[t]
        scale = torch.exp(exponent - max_exponent)
        wkv = (a * scale + wt * vt) / (b * scale + wt)

        # update state
        max_exponent = torch.max(exponent + w, kt)
        scale1 = torch.exp(exponent + w - max_exponent)
        scale2 = torch.exp(kt - max_exponent)
        a = scale1 * a + scale2 * vt
        b = scale1 * b + scale2
        exponent = max_exponent

        lst.append(wkv.unsqueeze(0))
    return torch.concat(lst, dim=0)


def make_mask(seq_len, device, dtype):
    a = torch.arange(seq_len, device=device, dtype=dtype).reshape(-1, 1)
    b = a.reshape(1, -1)
    c = a-b-1
    d = (c < -1)
    f = d.to(dtype=dtype).masked_fill_(d, float("-inf"))
    e = torch.eye(seq_len, device=device, dtype=dtype)
    return f.unsqueeze(2), c.unsqueeze(2), e.unsqueeze(2)


def parallel(
        key: torch.Tensor, value: torch.Tensor,
        u: torch.Tensor, w: torch.Tensor,
        left_down: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
    assert u.dim() == w.dim() == 1
    assert u.size(-1) == w.size(-1) == key.size(-1) == value.size(-1)
    seq_len = key.size(0)
    w = w.reshape(1, 1, -1)
    u = u.reshape(1, 1, -1)

    score = w*mask1+left_down+(u+w)*mask2
    key = key.reshape(1, seq_len, -1)
    score = score+key

    max_score = torch.max(score, dim=1, keepdim=True).values
    score = score-max_score

    exp = torch.exp(score)
    value = value.reshape(1, seq_len, -1)
    out = torch.sum(exp*value, dim=1)/torch.sum(exp, dim=1)
    return out


def main():
    '''
    展示了 rwkv 也可以并行化计算
    遗憾的是只在 cuda 上看到了加速，cpu 上反而更慢
    '''
    seq_len = 2000
    hidden_size = 256

    device = 'cuda'
    # device = 'cpu'
    dtype = torch.float32
    key = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    value = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    u = torch.randn(hidden_size, device=device, dtype=dtype)
    w = -torch.abs(torch.randn(hidden_size, device=device, dtype=dtype))

    left_down, mask1, mask2 = make_mask(seq_len, device=device, dtype=dtype)

    out1 = timer_func(iterative)(key, value, u, w, device, dtype)
    out2 = timer_func(parallel)(key, value, u, w, left_down, mask1, mask2)
    print((out1-out2).abs().max())


if __name__ == '__main__':
    main()
