from typing import Tuple, List
import numpy as np
from skimage import exposure

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def convert_image_to_patches(img: np.ndarray, patch_size: Tuple[int, ...], step_size: float) -> List:
    """
    """
    assert len(img.shape)==2 # Input image cannot be 3D.
    assert len(patch_size)==2 # If patch_size is bigger, the image cannot be converted
    img_shape = img.shape
    steps = _compute_steps_for_sliding_window(patch_size, img_shape, step_size)
    patches = []
    for x in steps[0]:
        low_x = x
        up_x = x + patch_size[0]
        for y in steps[1]:
            low_y = y
            up_y = y + patch_size[1]

            patches.append(img[low_x:up_x, low_y:up_y])

    return patches


def histogram_equalization(img: np.ndarray):
    return exposure.equalize_hist(img)*255


def adaptive_equalization(img: np.ndarray, clip_limit: float = 0.035):
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)*255
