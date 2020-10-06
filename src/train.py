import numpy as np
import cv2 as cv
import os
import re
import random
from utilities import projection
from glob import glob
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection  import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي','لا']
train_ratio = 0.8
script_path = os.getcwd()
classifiers = [ svm.LinearSVC(),
                MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100,), max_iter=1000),
                MLPClassifier(alpha=1e-5, hidden_layer_sizes=(200, 100,), max_iter=1000),
                GaussianNB()]

names = ['LinearSVM', '1L_NN', '2L_NN', 'Gaussian_Naive_Bayes']
skip = [1, 0, 1, 1]

width = 25
height = 25
dim = (width, height)


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


def binarize(char_img):
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    # _, binary_img = cv.threshold(word_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary_char = binary_img // 255

    return binary_char


def prepare_char(char_img):

    binary_char = binarize(char_img)

    try:
        char_box = bound_box(binary_char)
        resized = cv.resize(char_box, dim, interpolation = cv.INTER_AREA)
    except:
        pass

    return resized


def featurizer(char_img):

    flat_char = char_img.flatten()

    return flat_char


def read_data(limit=4000):

    X = []
    Y = []
    print("For each char")
    for char in tqdm(chars, total=len(chars)):

        folder = f'../Dataset/char_sample/{char}'
        char_paths =  glob(f'../Dataset/char_sample/{char}/*.png')

        if os.path.exists(folder):
            os.chdir(folder)

            print(f'\nReading images for char {char}')
            for char_path in tqdm(char_paths[:limit], total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                # X.append(char)
                X.append(feature_vector)
                Y.append(char)

            os.chdir(script_path)
            
    return X, Y


def train():

    X, Y = read_data()
    assert(len(X) == len(Y))

    X, Y = shuffle(X, Y)

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    scores = []
    for idx, clf in tqdm(enumerate(classifiers), desc='Classifiers'):

        if not skip[idx]:

            clf.fit(X_train, Y_train)
            score = clf.score(X_test, Y_test)
            scores.append(score)
            print(score)

            # Save the model
            destination = f'models'
            if not os.path.exists(destination):
                os.makedirs(destination)
            
            location = f'models/{names[idx]}.sav'
            pickle.dump(clf, open(location, 'wb'))


    with open('models/report.txt', 'w') as fo:
        for score, name in zip(scores, names):
            fo.writelines(f'Score of {name}: {score}\n')


def test(limit=3000):

    location = f'models/{names[0]}.sav'
    clf = pickle.load(open(location, 'rb'))
     
    X = []
    Y = []
    tot = 0
    for char in tqdm(chars, total=len(chars)):

        folder = f'../Dataset/char_sample/{char}'
        char_paths =  glob(f'../Dataset/char_sample/{char}/*.png')


        if os.path.exists(folder):
            os.chdir(folder)

            print(f'\nReading images for char {char}')
            tot += len(char_paths) - limit
            for char_path in tqdm(char_paths[limit:], total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                # X.append(char)
                X.append(feature_vector)
                Y.append(char)

            os.chdir(script_path)
    
    cnt = 0
    for x, y in zip(X, Y):

        c = clf.predict([x])[0]
        if c == y:
            cnt += 1


if __name__ == "__main__":

    train()
    # test()