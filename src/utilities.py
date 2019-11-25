import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def show_image(gray_img):
    plt.imshow(gray_img, 'gray')
    plt.show()


def save_image(img, folder, title):
    cv.imwrite(f'./{folder}/{title}.png', img)


def projection(gray_img, axis:str='horizontal'):
    """ Compute the horizontal or the vertical projection of a gray image """
    invert_img = cv.bitwise_not(gray_img)

    if axis == 'horizontal':
        projection_bins = np.sum(invert_img, 1)    
    elif axis == 'vertical':
        projection_bins = np.sum(invert_img, 0)    

    return projection_bins