import argparse
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from glob import glob

from tqdm import tqdm

from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
from utilities import read_image

log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

model_name = '2L_NN.sav'


def load_model():
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    with open(path, 'rb') as fin:
        return pickle.load(fin)


def run2(obj):
    word, line = obj
    model = load_model()

    # For each word in the image
    try:
        char_imgs = segment(line, word)
    except Exception as exc:
        # One bad word must not abort the whole image.
        log.warning('Skipping a word that failed to segment: %s: %s', type(exc).__name__, exc)
        return ''

    txt_word = ''
    skipped = 0
    # For each character in the word
    for char_img in char_imgs:
        ready_char = prepare_char(char_img)
        if ready_char is None:
            skipped += 1
            continue
        feature_vector = featurizer(ready_char)
        predicted_char = model.predict([feature_vector])[0]
        txt_word += predicted_char

    if skipped:
        log.debug('Dropped %d unusable character image(s) in a word', skipped)
    return txt_word


def run(image_path, output_dir):
    # Read test image
    full_image = read_image(image_path)
    if full_image is None:
        raise ValueError(f'Could not read image: {image_path}')

    # Start Timer
    before = time.time()
    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
    with mp.Pool(mp.cpu_count()) as pool:
        predicted_words = pool.map(run2, words)
    # Stop Timer
    after = time.time()

    # Rebuild the text line by line. Words of the same line share the same
    # `line` array object, which is how a new line is detected.
    lines_text = []
    current = []
    previous_line = None
    for (_, line), word_text in zip(words, predicted_words):
        if previous_line is not None and line is not previous_line:
            lines_text.append(' '.join(current))
            current = []
        current.append(word_text)
        previous_line = line
    if current:
        lines_text.append(' '.join(current))
    predicted_text = '\n'.join(lines_text)

    exc_time = after-before
    # Create file with the same name of the image
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    with open(os.path.join(output_dir, 'text', f'{img_name}.txt'), 'w', encoding='utf8') as fo:
        fo.write(predicted_text)

    return (img_name, exc_time)


def find_images(input_dir):
    return sorted(path for path in glob(os.path.join(input_dir, '*'))
                  if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Arabic OCR: images of typed text -> text files')
    parser.add_argument('--input', default=os.path.join(BASE_DIR, 'test'),
                        help='folder with the images (png, jpg, jpeg, bmp); default: src/test')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'output'),
                        help='folder for the results; default: src/output')
    parser.add_argument('-v', '--verbose', action='store_true', help='also log dropped characters')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format='%(levelname)s: %(message)s')

    output_dir = args.output
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)

    images_paths = find_images(args.input)
    if not images_paths:
        log.error('No images found in %s', args.input)
        return 1

    before = time.time()

    running_time = []
    failed = []

    for images_path in tqdm(images_paths, total=len(images_paths)):
        try:
            running_time.append(run(images_path, output_dir))
        except Exception as exc:
            log.error('Failed on %s: %s: %s', images_path, type(exc).__name__, exc)
            failed.append(images_path)

    running_time.sort()
    with open(os.path.join(output_dir, 'running_time.txt'), 'w') as r:
        for t in running_time:
            r.write(f'image#{t[0]}: {t[1]}\n')       # if no need for printing 'image#id'.

    after = time.time()
    print(f'total time to finish {len(running_time)} images:')
    print(after - before)
    if failed:
        print(f'{len(failed)} image(s) failed: {", ".join(failed)}')
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
