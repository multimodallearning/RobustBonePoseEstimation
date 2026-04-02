import torch
from skimage import measure, morphology
from kornia import morphology as k_morph
import numpy as np


def smart_cca(masks: torch.Tensor, min_abs_area: int, min_rel_to_biggest_area: float):
    """
    will only keep components that meet both conditions:
        * pixel count greater than min_abs_area
        * at least the define fraction size of the biggest one
    """
    assert masks.ndim == 4 and masks.dtype == torch.bool, 'expected mask to be boolean and of shape [B, C, H, W]'

    B, C, H, W = masks.shape
    cleaned_masks = torch.zeros(B * C, H, W, dtype=torch.bool)
    for i, mask in enumerate(masks.flatten(end_dim=1)):
        if not mask.any(): continue  # skip empty mask
        label_img = measure.label(mask.numpy())
        regions = measure.regionprops(label_img, cache=True)
        min_rel_area_thr = max([region.area for region in regions]) * min_rel_to_biggest_area
        area_thr = max(min_rel_area_thr, min_abs_area)
        for region in regions:
            if region.area > area_thr:
                cleaned_masks[i, region.coords[:, 0], region.coords[:, 1]] = True

    cleaned_masks = torch.unflatten(cleaned_masks, 0, [B, C])
    return cleaned_masks


def opening(masks: torch.Tensor, radius: int):
    kernel = torch.from_numpy(morphology.disk(radius, strict_radius=False))
    masks = k_morph.opening(masks.float(), kernel.float(), engine='convolution').bool()
    return masks


def opening_closing_cca(masks: torch.Tensor, radius: int):
    """
    performs opening and closing and only keep the largest component
    """
    kernel = torch.from_numpy(morphology.disk(radius, strict_radius=False)).float()
    masks = masks.float()
    masks = k_morph.opening(masks, kernel, engine='convolution')
    masks = k_morph.closing(masks, kernel, engine='convolution')

    # keep only largest component
    B, C, H, W = masks.shape
    cleaned_masks = torch.zeros(B * C, H, W, dtype=torch.bool)
    for i, mask in enumerate(masks.flatten(end_dim=1)):
        if not mask.any(): continue  # skip empty mask
        label_img = measure.label(mask.numpy())
        regions = measure.regionprops(label_img, cache=True)
        idx = np.argmax(np.asarray([region.area for region in regions]))
        largest_region = regions[idx]
        cleaned_masks[i, largest_region.coords[:, 0], largest_region.coords[:, 1]] = True
    cleaned_masks = torch.unflatten(cleaned_masks, 0, [B, C])

    return cleaned_masks


def skeletonize(masks: torch.Tensor):
    assert masks.ndim == 4 and masks.dtype == torch.bool, 'expected mask to be boolean and of shape [B, C, H, W]'
    B, C, H, W = masks.shape
    skeletone = torch.zeros(B * C, H, W, dtype=torch.bool)
    for i, mask in enumerate(masks.flatten(end_dim=1)):
        skeletone[i] = torch.from_numpy(morphology.skeletonize(mask.numpy()))
    skeletone = torch.unflatten(skeletone, 0, [B, C])

    return skeletone


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    img = torch.zeros(125, 125, dtype=torch.bool)
    img[:5, :5] = True
    img[50:100, 50:100] = True
    img[10:30, 10:30] = True

    # processed_img = smart_cca(img.unsqueeze(0).unsqueeze(0), 1 ** 2, .01)
    # processed_img = opening(img.unsqueeze(0).unsqueeze(0), 5)
    # processed_img = opening_closing_cca(img.unsqueeze(0).unsqueeze(0), 5)
    processed_img = skeletonize(img.unsqueeze(0).unsqueeze(0))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img.squeeze())
    axs[1].imshow(processed_img.squeeze())
    plt.show()
