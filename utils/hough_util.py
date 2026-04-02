import numpy as np
import torch
from skimage.transform import hough_line


class HoughLineExtractor:
    def __init__(self, img_hw: tuple[int, int], n_angle_samples: int = 360):
        self.hw = img_hw
        self.tested_angles = np.linspace(-np.pi / 2, np.pi / 2, n_angle_samples, endpoint=False)

    @staticmethod
    def convert_theta_to_v(theta: torch.Tensor):
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    def estimate(self, masks: torch.Tensor, convert_theta_to_v: bool = True):
        assert masks.dtype == torch.bool and masks.dim() == 4, 'Expect masks to be boolean and of shape [B,C,H,W]'
        assert masks.shape[-2:] == self.hw, 'Mismatch in image resolution'

        B, C = masks.shape[:2]
        theta = torch.empty(B * C)
        d = torch.empty_like(theta)
        for i, mask in enumerate(masks.flatten(0, 1)):
            if mask.any():
                h, theta_candidates, d_candidates = hough_line(mask.numpy(), theta=self.tested_angles)
                d_idx, theta_idx = np.unravel_index(np.argmax(h, axis=None), h.shape)
                theta[i] = float(theta_candidates[theta_idx])
                d[i] = float(d_candidates[d_idx])
            else:
                theta[i] = d[i] = torch.nan

        theta = torch.unflatten(theta, 0, [B, C])
        d = torch.unflatten(d, 0, [B, C])

        if convert_theta_to_v:
            return self.convert_theta_to_v(theta), d
        else:
            return theta, d

    def get_line_coords(self, masks: torch.Tensor):
        theta, d = self.estimate(masks, False)
        x = torch.linspace(0, self.hw[1], 100).repeat(*theta.shape, 1)
        y = torch.linspace(0, self.hw[0], 100).repeat(*theta.shape, 1)
        is_vertical = torch.sin(theta).abs() < 1e-6

        # bring in shape for broadcasting
        is_vertical = is_vertical.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        theta = theta.unsqueeze(-1)
        d = d.unsqueeze(-1)

        x_line = torch.where(
            is_vertical,
            (d / torch.cos(theta)).expand(-1, -1, x.shape[-1]),
            x
        )
        y_line = torch.where(
            is_vertical,
            y,
            (d - x * torch.cos(theta)) / torch.sin(theta),
        )

        return x_line, y_line
