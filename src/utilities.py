import os

import numpy as np
import cv2 as cv


def save_image(img, folder, title):
    cv.imwrite(f'./{folder}/{title}.png', img)


def read_image(path, flags=cv.IMREAD_COLOR):
    """Like cv.imread, but also works for non-ASCII paths (e.g. the Arabic
    folder names used for the character dataset) on Windows.

    Returns None when the file exists but cannot be decoded as an image.
    """

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv.imdecode(data, flags)


def write_image(path, img):
    """Like cv.imwrite, but also works for non-ASCII paths on Windows.

    The format is taken from the file extension (default: png).
    """

    extension = os.path.splitext(path)[1] or '.png'
    ok, buffer = cv.imencode(extension, img)
    if not ok:
        raise ValueError(f'Could not encode image for {path}')
    buffer.tofile(path)


def projection(gray_img, axis:str='horizontal'):
    """ Compute the horizontal or the vertical projection of a gray image """

    if axis == 'horizontal':
        projection_bins = np.sum(gray_img, 1).astype('int32')
    elif axis == 'vertical':
        projection_bins = np.sum(gray_img, 0).astype('int32')
    else:
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")

    return projection_bins
