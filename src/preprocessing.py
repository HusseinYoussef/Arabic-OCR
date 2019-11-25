import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def binary_otsus(image, filter:int=1):
    """ Binarize an image 0's and 255's using Otsu's Binarization """

    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    
    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


def deskew(binary_img):
    """ Rotate an image by some degrees to fix skewed images """
    invert_img = cv.bitwise_not(binary_img)

    # thresh = binary_otsus(gray_img, 0)

    coords = np.column_stack(np.where(invert_img > 0))
    angle = cv.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv.warpAffine(binary_img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    return rotated_img


def expand(gray_img):
    """ Expand the image by some white space horizontally in both directions"""

    (h, w) = gray_img.shape[:2]
    white_space = np.ones((h, 5)) * 255

    return np.block([white_space, gray_img, white_space])