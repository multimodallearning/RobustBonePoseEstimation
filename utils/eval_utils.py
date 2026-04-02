import torch


def theta(v1: torch.Tensor, v2: torch.Tensor):
    return torch.arccos(torch.abs(torch.linalg.vecdot(v1, v2))) * 180 / torch.pi


def l1_stats_robust(y_hat: torch.Tensor, y: torch.Tensor):
    assert y_hat.shape == y.shape
    l1 = torch.abs(y_hat - y)
    l1_nan_free = l1[~l1.isnan()]
    return l1_nan_free.mean().item(), l1_nan_free.std(
        unbiased=False).item(), l1.isnan().logical_not().float().mean().item()


def extract_alpha_beta(v: torch.Tensor, domain: str):
    match domain:
        case 'hip':
            alpha = theta(v[:, 0], v[:, 2])
            beta = theta(v[:, 0], v[:, 1])
        case 'wrist':
            alpha = theta(v[:, 0], v[:, 1])
            beta = theta(v[:, 2], v[:, 3])
        case 'autosafe':
            alpha = theta(v[:, 0], v[:, 1])
            beta = alpha  # dummy to work with evaluation scripts (will be canceled out by taking the mean)
        case _:
            raise ValueError('Unknown domain.')
    return alpha, beta


def extract_v(p1: torch.Tensor, p2: torch.Tensor):
    v = p2 - p1
    v_unit = v / torch.linalg.vector_norm(v, dim=-1)

    return v_unit
