import cv2 as cv
import os
from segmentation import extract_words
from character_segmentation import segment
from utilities import write_image
from glob import glob
from tqdm import tqdm


# sorted() so that images and texts pair up deterministically on every OS
img_paths = sorted(glob(os.path.join('..', 'Dataset', 'scanned', '*.png')))
txt_paths = sorted(glob(os.path.join('..', 'Dataset', 'text', '*.txt')))

CHARS_DIR = os.path.join('..', 'Dataset', 'chars')

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

    pairs = list(zip(img_paths[:limit], txt_paths[:limit]))
    skipped_images = 0

    print("Processing Images")
    for img_path, txt_path in tqdm(pairs, total=len(pairs)):

        assert os.path.splitext(os.path.basename(img_path))[0] == os.path.splitext(os.path.basename(txt_path))[0], \
            f'image/text mismatch: {img_path} vs {txt_path}'

        # Text Part of the image
        with open(txt_path, 'r', encoding='utf8') as fin:

            lines = fin.readlines()
            if not lines:
                print(f'\nSkipping {txt_path}: empty text file')
                skipped_images += 1
                continue
            line = lines[0].rstrip()

            txt_words = line.split()


        # Image Part
        img = cv.imread(img_path)
        if img is None:
            print(f'\nSkipping {img_path}: could not read image')
            skipped_images += 1
            continue
        img_words = extract_words(img)

        # If the segmentation found a different number of words than the text
        # has, pairing them up would give the characters after the mismatch
        # wrong labels, so the whole image is skipped.
        if len(img_words) != len(txt_words):
            print(f'\nSkipping {img_path}: {len(img_words)} words in the image, {len(txt_words)} in the text')
            skipped_images += 1
            continue

        # Get the words for the image
        for img_word, txt_word in zip(img_words, txt_words):

            # Get the text characters
            txt_chars = get_word_chars(txt_word)
            # Get the image characters
            line = img_word[1]
            word = img_word[0]

            try:
                img_chars = segment(line, word)
            except Exception as exc:
                print(f'\nSkipping a word in {img_path}: {type(exc).__name__}: {exc}')
                continue

            # Only words segmented into exactly as many pieces as they have
            # letters give reliable (image, label) pairs.
            if len(txt_chars) == len(img_chars):

                for img_char, txt_char in zip(img_chars, txt_chars):

                    if txt_char not in directory:
                        # Not one of the supported letters (digits, symbols, ...)
                        continue

                    number = directory[txt_char]

                    destination = os.path.join(CHARS_DIR, txt_char)
                    os.makedirs(destination, exist_ok=True)

                    write_image(os.path.join(destination, f'{number}.png'), img_char)

                    directory[txt_char] += 1

    print('\nDone')
    if skipped_images:
        print(f'{skipped_images} image(s) were skipped')


if __name__ == "__main__":
    prepare_dataset()
