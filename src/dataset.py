import cv2 as cv
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from segmentation import extract_words
from character_segmentation import segment
from glob import glob
from utilities import projection
from tqdm import tqdm

# def num(val):

#     numbers = re.compile('\d+')
#     match = numbers.search(val)
#     breakpoint()
#     return match.group(0)

img_paths = glob('../Dataset/scanned/*.png')
txt_paths = glob('../Dataset/text/*.txt')

# img_paths.sort(key = num)
# txt_paths.sort(key = num)

# breakpoint()


script_path = os.getcwd()
limit = 10

width = 25
height = 25
dim = (width, height)

directory = {}

directory["ا"] = 0
directory["ب"] = 0
directory["ت"] = 0
directory["ث"] = 0
directory["ج"] = 0
directory["ح"] = 0
directory["خ"] = 0
directory["د"] = 0
directory["ذ"] = 0
directory["ر"] = 0
directory["ز"] = 0
directory["س"] = 0
directory["ش"] = 0
directory["ص"] = 0
directory["ض"] = 0
directory["ط"] = 0
directory["ظ"] = 0
directory["ع"] = 0
directory["غ"] = 0
directory["ف"] = 0
directory["ق"] = 0
directory["ك"] = 0
directory["ل"] = 0
directory["م"] = 0
directory["ن"] = 0
directory["ه"] = 0
directory["و"] = 0
directory["ي"] = 0
directory["لا"] = 0


def bound_box(img_char):
    HP = projection(img_char, 'horizontal')
    VP = projection(img_char, 'vertical')

    top = -1
    down = -1
    left = -1
    right = -1

    i = 0
    while i < len(HP):
        if HP[i] != 0:
            top = i
            break
        i += 1

    i = len(HP)-1
    while i >= 0:
        if HP[i] != 0:
            down = i
            break
        i -= 1

    i = 0
    while i < len(VP):
        if VP[i] != 0:
            left = i
            break
        i += 1

    i = len(VP)-1
    while i >= 0:
        if VP[i] != 0:
            right = i
            break
        i -= 1

    return img_char[top:down+1, left:right+1]


def check_lamAlf(word, idx):

    if idx != len(word)-1 and word[idx] == 'ل':
        if word[idx+1] == 'ا':
            return True

    return False


def get_word_chars(word):

    i = 0
    chars = []
    while i < len(word):
        if check_lamAlf(word, i):
            chars.append(word[i:i+2])
            i += 2
        else:
            chars.append(word[i])
            i += 1

    return chars


print("Processing Images")
counter = 0
for img_path, txt_path in tqdm(zip(img_paths, txt_paths), total=len(img_paths)):

    if counter > limit:
        break

    test = 'capr2'
    txt_path = f'../Dataset/text/{test}.txt'
    img_path = f'../Dataset/scanned/{test}.png'

    # Text Part of the image
    with open(txt_path, 'r', encoding='utf8') as fin:

        lines = fin.readlines()
        line = lines[0].rstrip()

        # assert len(lines) == 1, 'Number of lines != 1'

        # Remove punctuation
        # line = re.sub(r'[\d:,;.\[\]()]+', '', line)
        txt_words = line.split()
        # print(f'Number of text words = {len(txt_words)}')

    
    # Image Part
    img = cv.imread(img_path)
    img_words = extract_words(img)

    # breakpoint()

    lst = []
    error = 0
    
    # Get the words for the image
    for img_word, txt_word in tqdm(zip(img_words, txt_words), total=len(txt_words)):
        
        # Get the text characters
        txt_chars = get_word_chars(txt_word)
        # Get the image characters
        line = img_word[1]
        word = img_word[0]

        img_chars = segment(line, word)
        
        # breakpoint()
        if len(txt_chars) == len(img_chars):
            
            for img_char, txt_char in zip(img_chars, txt_chars):
                
                number = directory[txt_char]

                destination = f'../Dataset/chars/{txt_char}'
                if not os.path.exists(destination):
                    os.makedirs(destination)
                
                char_box = bound_box(img_char)
                resized = cv.resize(char_box, dim, interpolation = cv.INTER_AREA)
                
                # plt.imshow(char_box, 'gray')
                # plt.show()
                
                os.chdir(destination)
                cv.imwrite(f'{number}.png', resized)
                os.chdir(script_path)

                directory[txt_char] += 1

        else:
            error += 1

    print(f'\nAcc: {100-(error*100/len(img_words))}')
    counter += 1
    # breakpoint()

print('\nDone')
