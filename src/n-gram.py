import nltk
import numpy as np
import random
import string
from glob import glob
import bs4 as bs
import re
from tqdm import tqdm


if __name__ == "__main__":
    txt_paths = glob('../Dataset/text/*.txt')
    counter = 0
    limit = 10
    full_text = ''
    for txt_path in tqdm(txt_paths, total=len(txt_paths)):
        if counter > limit:
            break
        with open(txt_path, 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            line = lines[0].rstrip()
            full_text += line
            full_text += " "
    
    words_tokens = full_text.split()
    ngrams = {}
    words = 3
    for i in range(len(words_tokens)-words):
        seq = ' '.join(words_tokens[i:i+words])
        if seq not in ngrams.keys():
            ngrams[seq] = []
        ngrams[seq].append(words_tokens[i+words])


