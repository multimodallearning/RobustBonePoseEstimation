import torch
from clearml import Task

from model.line_seg_models import LineUNet
from model.heatmap_models import HeatMapUNet
from utils import ransac_util, hough_util, false_positiv_reduction, eval_utils


class SegLinePredictor:
    def __init__(self, model_clearml_id: str, line_extraction: str, ransac_k: float = 2.5,
                 mask_cleaning: str = None, morph_radius: int = 2, cca_min_abs_area: int = None,
                 cca_min_relation_biggest_area: float = None):
        # load model
        task = Task.get_task(model_clearml_id)
        ckpt = task.artifacts['best.ckpt'].get_local_copy()
        self.model = LineUNet.load_from_checkpoint(ckpt, 'cpu').eval()
        self.params = task.get_parameters_as_dict(cast=True)['Args']
        match self.params['fit.data.class_path'].split('.')[-1]:
            case 'USHipLineDataModule' | 'USHipDataModule':
                self.ds = 'hip'
                self.hw = (384, 320)
            case 'FractureAngleLineDataModule':
                self.ds = 'wrist'
                self.hw = (384, 224)
            case 'AutoSafeDataModule':
                self.ds = 'autosafe'
                self.hw = (320, 320)
            case _:
                raise NotImplementedError('Yet unknown dataset.')

        # line extractor
        match line_extraction.lower():
            case 'pca':
                self.line_extractor = ransac_util.PCALineModel(self.hw)
            case 'ransac':
                self.line_extractor = ransac_util.PointDirectionRansacLineExtractor(self.hw, 10, k=ransac_k)
            case 'hough':
                self.line_extractor = hough_util.HoughLineExtractor(self.hw)
            case _:
                raise ValueError('Unknown line extractor')

        # preprocessing
        if (mask_cleaning is None) or (mask_cleaning.lower() == 'none'):
            self.preproc = lambda m: m
        else:
            match mask_cleaning.lower():
                case 'cca':
                    self.preproc = lambda m: false_positiv_reduction.smart_cca(m, cca_min_abs_area,
                                                                               cca_min_relation_biggest_area)
                case 'opening':
                    self.preproc = lambda m: false_positiv_reduction.opening(m, morph_radius)
                case 'cv_pipeline':
                    self.preproc = lambda m: false_positiv_reduction.opening_closing_cca(m, morph_radius)
                case 'skeletonize':
                    self.preproc = lambda m: false_positiv_reduction.skeletonize(m)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        y_hat = self.model(x) > 0
        y_hat_cleaned = self.preproc(y_hat)
        if isinstance(self.line_extractor, hough_util.HoughLineExtractor):
            v_hat, origin = self.line_extractor.estimate(y_hat_cleaned, convert_theta_to_v=True)
        else:
            v_hat, origin = self.line_extractor.estimate(y_hat_cleaned)

        return v_hat, origin

    def predict_angle(self, x: torch.Tensor):
        v_hat, _ = self.predict(x)
        alpha_hat, beta_hat = eval_utils.extract_alpha_beta(v_hat, self.ds)

        return alpha_hat, beta_hat

    @torch.inference_mode()
    def predict_masks(self, x: torch.Tensor, apply_mask_cleaning: bool = False):
        y_hat = self.model(x) > 0
        if apply_mask_cleaning:
            y_hat = self.preproc(y_hat)
        return y_hat

    @torch.inference_mode()
    def predict_for_plotting(self, x: torch.Tensor):
        y_hat = self.model(x) > 0
        y_hat_cleaned = self.preproc(y_hat)
        x_coord_hat, y_coord_hat = self.line_extractor.get_line_coords(y_hat_cleaned)

        return y_hat, y_hat_cleaned, (x_coord_hat, y_coord_hat)

    @property
    def pca_line_model(self):
        return ransac_util.PCALineModel(self.hw)

    @property
    def hough_line_model(self):
        return hough_util.HoughLineExtractor(self.hw)


class KptsLinePredictor:
    def __init__(self, model_clearml_id: str):
        # load model
        task = Task.get_task(model_clearml_id)
        ckpt = task.artifacts['best.ckpt'].get_local_copy()
        self.model = HeatMapUNet.load_from_checkpoint(ckpt, 'cpu').eval()
        self.params = task.get_parameters_as_dict(cast=True)['Args']
        match self.params['fit.data.class_path'].split('.')[-1]:
            case 'USHipLineDataModule' | 'USHipDataModule':
                self.ds = 'hip'
            case 'FractureAngleKeyPointDataModule':
                self.ds = 'wrist'
            case 'AutoSafeDataModule':
                self.ds = 'autosafe'
            case _:
                raise NotImplementedError('Yet unknown dataset.')

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        y_hat = self.model(x)
        kpts_head = self.model.extract_kpts_from_heatmap(y_hat, 4)

        return kpts_head

    def predict_v(self, x: torch.Tensor):
        return self.convert_to_v(self.predict(x), self.ds)

    @staticmethod
    def convert_to_v(kpts: torch.Tensor, domain):
        assert kpts.ndim == 3, 'kpts need to be of shape [B, K, 2]'
        match domain:
            case 'wrist':
                v = torch.stack([
                    eval_utils.extract_v(kpts[:, 0], kpts[:, 1]),
                    eval_utils.extract_v(kpts[:, 2], kpts[:, 3]),
                    eval_utils.extract_v(kpts[:, 4], kpts[:, 5]),
                    eval_utils.extract_v(kpts[:, 6], kpts[:, 7])
                ], 1)
                p0 = kpts[:, [0, 2, 4, 6]]
            case 'hip':
                v = torch.stack([
                    eval_utils.extract_v(kpts[:, 0], kpts[:, 1]),
                    eval_utils.extract_v(kpts[:, 2], kpts[:, 3]),
                    eval_utils.extract_v(kpts[:, 4], kpts[:, 5])
                ], 1)
                p0 = kpts[:, [1, 2, 4]]
            case 'autosafe':
                v = torch.stack([
                    eval_utils.extract_v(kpts[:, 0], kpts[:, 1]),
                    eval_utils.extract_v(kpts[:, 2], kpts[:, 3]),
                ], 1)
                p0 = kpts[:, [1, 2]]
            case _:
                raise ValueError('Unknown domain.')
        return v, p0
