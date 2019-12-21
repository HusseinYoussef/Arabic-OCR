from glob import glob
import cv2 as cv
from character_segmentation import segment
from segmentation import extract_words
import pickle
from train import prepare_char,featurizer
import os
import time


os.mkdir('output')
os.mkdir('output/text')
location = 'LinearSVM.sav'
model = pickle.load(open(location,'rb'))
time_file = open('output/running_time.txt','w+')
images_paths = glob('test/*.png')
counter = 0
for image_path in images_paths:
    counter+=1
    f = open(f'output/text/test_{counter}.txt' , 'w+' , encoding='utf8')
    full_image = cv.imread(image_path)
    before = time.time()
    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
    for word,line in words:
        chars = segment(line,word)
        for char in chars:  
            prepared_char = prepare_char(char)
            features = featurizer(prepared_char)
            predicted_char = model.predict([features])
            f.write(predicted_char[0])
        f.write(' ')
    after = time.time()
    time_file.write(str(after-before) + '\n')
    f.close()

time_file.close()