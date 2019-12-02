import numpy as np
import cv2 as cv

'''
Extract features from character image
'''

def get_number_loops(character_img):

    contours, hierarchy = cv.findContours(character_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Number of internal contours
    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1
    
    return cnt


def get_connected_components(character_img, c):

    # c = 4 or 8
    components, labels= cv.connectedComponents(character_img, connectivity=c)

    # Discard background component
    return components - 1