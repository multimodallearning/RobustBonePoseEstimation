import torch
from sklearn.linear_model import RANSACRegressor
from abc import abstractmethod, ABC
from skimage.measure import ransac, LineModelND
from skimage._shared.utils import FailedEstimationAccessError


class BaseLineModelWithCoordinates(ABC):
    def __init__(self, img_hw: tuple[int, int]):
        self.coord_grid = torch.meshgrid([torch.arange(img_hw[0]), torch.arange(img_hw[1])], indexing='ij')
        self.coord_grid = torch.stack(self.coord_grid, -1)
        self.line_y = torch.linspace(0, img_hw[0], img_hw[0])

    @abstractmethod
    def estimate(self, masks: torch.Tensor):
        pass

    @abstractmethod
    def get_line_coords(self, masks: torch.Tensor):
        pass


class MxBRansacLineExtractor:
    def __init__(self, img_hw: tuple[int, int], residual_threshold: float = None, min_samples: int = 4):
        assert img_hw[0] >= img_hw[1], 'Expect image to have an portrait aspect ratio'
        self.ransac_kwargs = dict(residual_threshold=residual_threshold, min_samples=min_samples)
        self.coord_grid = torch.meshgrid([torch.arange(img_hw[0]), torch.arange(img_hw[1])], indexing='ij')
        self.coord_grid = torch.stack(self.coord_grid, -1)
        self.line_y = torch.linspace(0, img_hw[0], img_hw[0])

    def estimate(self, masks: torch.Tensor):
        assert masks.dtype == torch.bool and masks.dim() == 4, 'Expect masks to be boolean and of shape [B,C,H,W]'
        assert masks.shape[-2:] == self.coord_grid.shape[:2], 'Mismatch in image resolution'

        B, C = masks.shape[:2]
        coef = torch.empty(B * C)
        intercept = torch.empty_like(coef)
        for i, mask in enumerate(masks.flatten(0, 1)):
            coords = self.coord_grid[mask]
            try:
                ransac = RANSACRegressor(**self.ransac_kwargs)
                ransac.fit(coords[:, 0].unsqueeze(-1).numpy(), coords[:, 1].unsqueeze(-1).numpy())
                coef[i] = float(ransac.estimator_.coef_.squeeze())
                intercept[i] = float(ransac.estimator_.intercept_.squeeze())
            except ValueError:
                coef[i] = torch.nan
                intercept[i] = torch.nan
        coef = torch.unflatten(coef, 0, [B, C])
        intercept = torch.unflatten(intercept, 0, [B, C])

        return coef, intercept

    def get_line_coords(self, masks: torch.Tensor):
        coef, intercept = self.estimate(masks)
        y = self.line_y.view(1, 1, -1).expand(*masks.shape[:2], -1)
        x = coef.unsqueeze(-1) * y + intercept.unsqueeze(-1)

        return x, y


class PointDirectionRansacLineExtractor(BaseLineModelWithCoordinates):
    def __init__(self, img_hw: tuple[int, int], residual_threshold: float=None, k: float = 2.5):
        super().__init__(img_hw)
        del self.line_y
        self.res_thr = residual_threshold
        self.k = k

    @staticmethod
    def mad(values: torch.Tensor):
        return torch.median(abs(values - torch.median(values)))

    def estimate(self, masks: torch.Tensor):
        assert masks.dtype == torch.bool and masks.dim() == 4, 'Expect masks to be boolean and of shape [B,C,H,W]'
        assert masks.shape[-2:] == self.coord_grid.shape[:2], 'Mismatch in image resolution'

        B, C = masks.shape[:2]
        direction = torch.empty(B * C, 2)
        origin = torch.empty_like(direction)
        for i, mask in enumerate(masks.flatten(0, 1)):
            coords = self.coord_grid[mask].flip(-1).float()
            try:
                if self.k is None:
                    curr_res_thr = self.res_thr
                else:
                    loose_model, _ = ransac(coords, LineModelND, 2, self.res_thr, stop_probability=0.99)
                    residuals = torch.from_numpy(loose_model.residuals(coords))
                    sigma = 1.4826 * self.mad(residuals).item()
                    curr_res_thr = sigma * self.k

                robust_line_model, _ = ransac(coords, LineModelND, 2, curr_res_thr, stop_probability=0.99)
                if robust_line_model is None:
                    raise FailedEstimationAccessError

                direction[i] = robust_line_model.direction
                origin[i] = robust_line_model.origin
            except (FailedEstimationAccessError, ValueError):
                direction[i] = torch.nan
                origin[i] = torch.nan
        direction = torch.unflatten(direction, 0, [B, C])
        origin = torch.unflatten(origin, 0, [B, C])

        return direction, origin

    def get_line_coords(self, masks: torch.Tensor):
        v, p0 = self.estimate(masks)
        t = torch.linspace(-100, 100, 100)
        line_xy = p0.unsqueeze(-2) + t.view(1, 1, -1, 1) * v.unsqueeze(-2)

        return line_xy[..., 0], line_xy[..., 1]


class PCALineModel(PointDirectionRansacLineExtractor):
    def __init__(self, img_hw: tuple[int, int]):
        super().__init__(img_hw, 0, 0)

    def estimate(self, masks: torch.Tensor, return_max_residuals: bool = False):
        assert masks.dtype == torch.bool and masks.dim() == 4, 'Expect masks to be boolean and of shape [B,C,H,W]'
        assert masks.shape[-2:] == self.coord_grid.shape[:2], 'Mismatch in image resolution'

        B, C = masks.shape[:2]
        direction = torch.empty(B * C, 2)
        origin = torch.empty_like(direction)
        residual = torch.empty(B * C)
        for i, mask in enumerate(masks.flatten(0, 1)):
            coords = self.coord_grid[mask].flip(-1).float()
            try:
                line_model = LineModelND.from_estimate(coords)
                direction[i] = line_model.direction
                origin[i] = line_model.origin
                residual[i] = float(line_model.residuals(coords).max())
            except FailedEstimationAccessError:
                direction[i] = torch.nan
                origin[i] = torch.nan
                residual[i] = torch.nan
        direction = torch.unflatten(direction, 0, [B, C])
        origin = torch.unflatten(origin, 0, [B, C])
        residual = torch.unflatten(residual, 0, [B, C])

        if return_max_residuals:
            return direction, origin, residual
        else:
            return direction, origin
