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
classifiers = [svm.LinearSVC(), MLPClassifier(alpha=1e-5, hidden_layer_sizes=(1, 100)), GaussianNB()]
names = ['LinearSVM', 'Neural_Nets_1', 'Gaussian_Naive_Bayes']
skip = [0, 1, 1]

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


def prepare_char(char_img):
    _, binary_img = cv.threshold(char_img, 127, 255, cv.THRESH_BINARY)
    # _, binary_img = cv.threshold(word_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binary_char = binary_img // 255

    char_box = bound_box(binary_char)
    resized = cv.resize(char_box, dim, interpolation = cv.INTER_AREA)

    
    # breakpoint()
    return resized


def featurizer(char_img):

    flat_char = char_img.flatten()

    return flat_char


def read_data():

    X = []
    Y = []
    print("For each char")
    for char in tqdm(chars, total=len(chars)):

        folder = f'../Dataset/char_sample/{char}'
        char_paths =  glob(f'../Dataset/char_sample/{char}/*.png')

        # breakpoint()
        if os.path.exists(folder):
            os.chdir(folder)

            print(f'Reading images for char {char}')
            for char_path in tqdm(char_paths, total=len(char_paths)):
                num = re.findall(r'\d+', char_path)[0]
                # breakpoint()
                char_img = cv.imread(f'{num}.png', 0)
                ready_char = prepare_char(char_img)
                feature_vector = featurizer(ready_char)
                X.append(char)
                # X.append(feature_vector)
                Y.append(char)
                # breakpoint()

            os.chdir(script_path)
            
    return X, Y


def train():

    X, Y = read_data()
    assert(len(X) == len(Y))

    X, Y = shuffle(X, Y)

    for x, y in zip(X, Y):
        if x != y:
            breakpoint()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    breakpoint()
    for idx, clf in enumerate(classifiers):

        if not skip[idx]:

            clf.fit(X_train, Y_train)
            print(clf.score(X_test, Y_test))

            # Save the model
            location = f'models/{names[idx]}.sav'
            
    # indices = [i for i in range(len(X))]
    # random.shuffle(indices)

    # X_data = X[indices]
    # Y_labels = Y[indices]

    # num_train = int((80/100) * len(X))

    # X_train = X_data[:num_train]
    # Y_train = Y_labels[:num_train]
    
    # X_test = X_data[num_train:]
    # Y_test = Y_labels[num_train:]


train()