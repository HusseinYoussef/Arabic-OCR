import cv2 as cv
import os
import re
from segmentation import extract_words
from character_segmentation import segment
from glob import glob
from utilities import projection
from tqdm import tqdm


img_paths = glob('..\\Dataset\\scanned/*.png')
txt_paths = glob('..\\Dataset\\text/*.txt')

script_path = os.getcwd()
width = 25
height = 25
dim = (width, height)

directory = {}
chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي','لا']

for char in chars:
    directory[char] = 0
    

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


def prepare_dataset(limit=1000):

    print("Processing Images")
    for img_path, txt_path in tqdm(zip(img_paths[:limit], txt_paths[:limit]), total=len(img_paths)):

        assert(img_path.split('\\')[-1].split('.')[0] == txt_path.split('\\')[-1].split('.')[0])

        # Text Part of the image
        with open(txt_path, 'r', encoding='utf8') as fin:

            lines = fin.readlines()
            line = lines[0].rstrip()

            txt_words = line.split()
     
        
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
                        
                    os.chdir(destination)
                    cv.imwrite(f'{number}.png', img_char)
                    os.chdir(script_path)

                    directory[txt_char] += 1

            else:
                error += 1
                # lst.append(txt_word)

        # print(f'\nAcc: {100-(error*100/len(img_words))}')

    print('\nDone')


if __name__ == "__main__":
    prepare_dataset()