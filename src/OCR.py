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


model_name = '2L_NN.sav'
def load_model():

    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model
        

def main():

    print("Loading Model...")
    model = load_model()
    print("Done loading the model.")

    destination = 'output/text'
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    images_paths = glob('test/*.png')

    with open('output/running_time.txt', 'w') as time_file:
        for image_path in tqdm(images_paths, total=len(images_paths)):
            
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

            time_file.writelines(f'{after-before}\n')
            # Create file with the same name of the image
            img_name = image_path.split('\\')[1].split('.')[0]
            with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
                fo.writelines(predicted_text)
            

if __name__ == "__main__":

    main()