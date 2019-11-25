import cv2 as cv
from preprocessing import binary_otsus, deskew
from utilities import projection, save_image
from glob import glob


def preprocess(image):

    # Maybe we end up using only gray level image.
    binary_img = binary_otsus(image, 0)
    clean_img = deskew(binary_img)
    save_image(clean_img, 'res')

    return clean_img


def projection_segmentation(clean_img, axis, cut=2, threshold=7):
    
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


# Line Segmentation
#----------------------------------------------------------------------------------------
def line_horizontal_projection(image, cut=2, threshold=7):

    # Preprocess input image
    clean_img = preprocess(image)

    # Segmentation    
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut, threshold=threshold)

    return lines


# Word Segmentation
#----------------------------------------------------------------------------------------
def word_vertical_projection(line_images:list, cut=2, threshold=7):
    
    lines_words = []
    for line_image in line_images:
        line_words = projection_segmentation(line_image, axis='vertical', cut=cut, threshold=threshold)
        lines_words.append(line_words)

    return lines_words


if __name__ == "__main__":
    # paths = glob('../Dataset/scanned/*.png')
    # breakpoint()
    image = cv.imread('../src/lines/line2.png', 0)
    # lines = line_horizontal_projection(image)
    
    # for idx, line in enumerate(lines):
    #     save_image(line,'lines', f'line{idx}')

    lines_words = word_vertical_projection([image])
    # breakpoint()
    for line_words in lines_words:
        for idx, word in enumerate(line_words):
            save_image(word, 'words', f'word{idx}')

    # breakpoint()
    # print(len(lines))