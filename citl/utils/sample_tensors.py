import torch


def sample_tensors(t1, t2, percentage=1.0):
    if t1.numel() == 0:
        return t1, t2
    count = t1.shape[0]
    assert t1.shape[0] == t2.numel()
    num_samples = int(percentage * count)
    num_samples = min(num_samples, count)
    sample_idx = torch.randperm(num_samples)[:num_samples]
    return t1[sample_idx], t2[sample_idx]


__all__ = ["sample_tensors"]
