import cv2 as cv
import os
import time
from tqdm import tqdm
from glob import glob
from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp

mutex = mp.Lock()

model_name = '2L_NN.sav'
def load_model():
    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model
        
def run(image_path):
    # print('here')
    model = load_model()
    # Read test image
    full_image = cv.imread(image_path)
    predicted_text = ''

    # Start Timer
    before = time.time()
    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]

    # For each word in the image
    for word, line in words:
        char_imgs = segment(line, word)
        txt_word = ''

        # For each character in the word
        for char_img in char_imgs:
            try:
                ready_char = prepare_char(char_img)
            except:
                # plt.imshow(word, 'gray')
                # plt.show()
                # breakpoint()
                continue
            feature_vector = featurizer(ready_char)
            predicted_char = model.predict([feature_vector])[0]
            txt_word += predicted_char

        if len(predicted_text):
            predicted_text += ' '
        predicted_text += txt_word

    after = time.time()
    # Stop Timer

    exc_time = after-before
    # Create file with the same name of the image
    img_name = image_path.split('\\')[1].split('.')[0]
    # img_idx = int(img_name.split('_')[1])         # the valid one for testing day.
    img_idx = int(''.join(i for i in img_name if i.isdigit()))
    with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
        fo.writelines(predicted_text)

    global mutex
    mutex.acquire()
    with open('output/running_time.txt', 'a') as fo:
        fo.write(f'{img_idx}: {exc_time:.2f}s\n')
    mutex.release()
    

# The following function is optional.
def sort_output():
    RUN = []
    with open('output/running_time.txt', 'r') as r:
        for line in r:
            RUN.append((int(line.split(':')[0]), line.split(':')[1]))
    RUN.sort()
    with open('output/running_time.txt', 'w') as r:
        for t in RUN:
            r.writelines(f'image#{t[0]}: {t[1]}')       # if no need for printing 'image#id'.


if __name__ == "__main__":

    #Clear the old data in running_time.txt
    open('output/running_time.txt', 'w').close()

    destination = 'output/text'
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    images_paths = glob('test/*.png')

    before = time.time()

    pool = mp.Pool(mp.cpu_count())

    # # Method1
    # for image_path in images_paths:
    #     pool.apply_async(run,[image_path])

    # Method2
    for _ in tqdm(pool.imap_unordered(run,images_paths), total=len(images_paths)):
        pass

    pool.close()
    pool.join()
    after = time.time()
    print(f'total time to finish {len(images_paths)} images:')
    print(after - before)
    sort_output()
