import cv2 as cv
import os
import time
from tqdm import tqdm
from glob import glob
from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
import pickle
import multiprocessing as mp

model_name = '2L_NN.sav'
def load_model():
    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model
        
def run2(obj):
    word, line = obj
    model = load_model()
    # For each word in the image
    char_imgs = segment(line, word)
    txt_word = ''
    # For each character in the word
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)
        except:
            # breakpoint()
            continue
        feature_vector = featurizer(ready_char)
        predicted_char = model.predict([feature_vector])[0]
        txt_word += predicted_char
    return txt_word


def run(image_path):
    # Read test image
    full_image = cv.imread(image_path)
    predicted_text = ''

    # Start Timer
    before = time.time()
    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
    pool = mp.Pool(mp.cpu_count())
    predicted_words = pool.map(run2, words)
    pool.close()
    pool.join()
    # Stop Timer
    after = time.time()

    # append in the total string.
    for word in predicted_words:
        predicted_text += word
        predicted_text += ' '

    exc_time = after-before
    # Create file with the same name of the image
    img_name = image_path.split('\\')[1].split('.')[0]

    with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
        fo.writelines(predicted_text)

    return (img_name, exc_time)


if __name__ == "__main__":

    #Clear the old data in running_time.txt
    if not os.path.exists('output'):
        os.mkdir('output')
    open('output/running_time.txt', 'w').close()

    destination = 'output/text'
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    types = ['png', 'jpg', 'bmp']
    images_paths = []
    for t in types:
        images_paths.extend(glob(f'test/*.{t}'))
    before = time.time()

    # pool = mp.Pool(mp.cpu_count())

    # # Method1
    # for image_path in images_paths:
    #     pool.apply_async(run,[image_path])

    # Method2
    # for _ in tqdm(pool.imap_unordered(run, images_paths), total=len(images_paths)):
    #     pass

    running_time = []

    for images_path in tqdm(images_paths,total=len(images_paths)):
        running_time.append(run(images_path))

    running_time.sort()
    with open('output/running_time.txt', 'w') as r:
        for t in running_time:
            r.writelines(f'image#{t[0]}: {t[1]}\n')       # if no need for printing 'image#id'.

    # pool.close()
    # pool.join()
    after = time.time()
    print(f'total time to finish {len(images_paths)} images:')
    print(after - before)
