import numpy as np
import cv2 as cv
import os
import re
import sys
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
    """Crop a character image to its bounding box and resize it to `dim`.

    Returns None when the image is empty or cannot be processed, so callers
    must skip None results instead of relying on an exception.
    """

    if char_img is None or char_img.size == 0:
        return None

    binary_char = binarize(char_img)

    # An all-black crop has no ink, hence no bounding box.
    if not binary_char.any():
        return None

    try:
        char_box = bound_box(binary_char)
        return cv.resize(char_box, dim, interpolation=cv.INTER_AREA)
    except cv.error:
        return None


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
                if ready_char is None:
                    print(f'Skipping unusable image: {char_path}')
                    continue
                feature_vector = featurizer(ready_char)
                # X.append(char)
                X.append(feature_vector)
                Y.append(char)

            os.chdir(script_path)

    return X, Y


def read_report(path='models/report.txt'):
    """Return {model name: score line value} from an existing report."""

    scores = {}
    if os.path.exists(path):
        with open(path, encoding='utf8') as fin:
            for line in fin:
                match = re.match(r'Score of (\S+): (\S+)', line)
                if match:
                    scores[match.group(1)] = match.group(2)
    return scores


def train(selected=None):
    """Train the classifiers listed in `selected` (default: all of `names`)."""

    selected = list(names) if not selected else selected
    unknown = [n for n in selected if n not in names]
    if unknown:
        raise ValueError(f'Unknown model(s) {unknown}; choose from {names}')

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

    # Keep scores of models that were not retrained this time.
    scores = read_report()
    for name, clf in tqdm(list(zip(names, classifiers)), desc='Classifiers'):

        if name not in selected:
            continue

        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        scores[name] = str(score)
        print(f'{name}: {score}')

        # Save the model
        destination = 'models'
        if not os.path.exists(destination):
            os.makedirs(destination)

        with open(f'models/{name}.sav', 'wb') as fo:
            pickle.dump(clf, fo)

    with open('models/report.txt', 'w') as fo:
        for name in names:
            if name in scores:
                fo.write(f'Score of {name}: {scores[name]}\n')


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
                if ready_char is None:
                    continue
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

    # Usage: python train.py [MODEL ...]   (default: all models in `names`)
    train(sys.argv[1:])
    # test()