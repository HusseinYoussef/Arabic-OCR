import numpy as np
import cv2 as cv
from preprocessing import binary_otsus, deskew
from utilities import projection, save_image
from glob import glob
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin


def preprocess(image):

    # Maybe we end up using only gray level image.
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = cv.bitwise_not(gray_img)

    # cv.imwrite('origin.png', gray_img)
    binary_img = binary_otsus(gray_img, 0)
    deskewed_img = deskew(binary_img)


    # binary_img = binary_otsus(deskewed_img, 0)
    # breakpoint()

    # Visualize
    # cv.imwrite('output.png', deskewed_img)

    # breakpoint()
    return deskewed_img


def projection_segmentation(clean_img, axis, cut=2):
    
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):

        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    segments.append(clean_img[max(start-1, 0):idx, :])
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
    
    return segments


def baseline_detection(line_binary_image):

    # Perform morphological thinning
    thinned_img = thin(line_binary_image)

    # Compute the Horizontal Projection (HP) for the thinned image
    HP = projection(thinned_img, 'horizontal')

    # baseline index is the index corresponding to the peak value
    baseline_idx = np.where(HP == np.amax(HP))[0][0]
    
    # line_binary_image[baseline_idx, :] = 255
    # cv.imwrite('line.png', line_binary_image)
    # breakpoint()

    return baseline_idx


def maximum_transitions(line_binary_image, baseline_idx):

    max_transitions = 0
    max_transitions_idx = baseline_idx
    line_idx = baseline_idx

    while line_idx >= 0:

        current_transitions = 0
        flag = 0

        horizontal_line = line_binary_image[line_idx, :]
        for pixel in reversed(horizontal_line):

            if pixel == 255 and flag == 0:
                current_transitions += 1
                flag = 1
            elif pixel == 0 and flag == 1:
                # current_transitions += 1
                flag = 0
        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            max_transitions_idx = line_idx

        line_idx -= 1
    
    line_binary_image[baseline_idx, :] = 255
    line_binary_image[max_transitions_idx, :] = 255
    cv.imwrite('line.png', line_binary_image)
    breakpoint()

    return max_transitions_idx


# Line Segmentation
#----------------------------------------------------------------------------------------
def line_horizontal_projection(image, cut=2):

    # Preprocess input image
    clean_img = preprocess(image)

    # Segmentation    
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)

    return lines


# Word Segmentation
#----------------------------------------------------------------------------------------
def word_vertical_projection(line_images:list, cut=3):
    
    lines_words = []
    for line_image in line_images:
        line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
        lines_words.append(line_words)

    return lines_words


if __name__ == "__main__":
    pass
    # paths = glob('../Dataset/scanned/*.png')
    # breakpoint()

    img = cv.imread('../Dataset/scanned/capr2.png')
    lines = line_horizontal_projection(img)
    for idx, line in enumerate(lines):
        save_image(line, 'lines', f'line{idx}')

    words = word_vertical_projection([lines[6]])
    for idx, word in enumerate(words[0]):
        save_image(word, 'words', f'word{idx}')

    # baseline = baseline_detection(lines[3])
    # maximum_transitions(lines[3], baseline)