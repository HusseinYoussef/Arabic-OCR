# **Arabic OCR**

- OCR system for Arabic language that converts images of typed text to machine-encoded text.<br>
- The system currently supports only letters (29 letters) ا-ى , لا.
- The system aims to solve a simpler problem of OCR with images that contain only Arabic characters (check the dataset link below to see a sample of the images).

## Setup

Install python then run this command:

```shell
pip install -r requirements.txt
```

## Run

1. Put the images in src/test directory
2. Go to src directory and run the following command
   ```shell
   python OCR.py
   ```
3. Output folder will be created with:
   - text folder which has text files corresponding to the images.
   - running_time file which has the time taken to process each image.

## Pipeline

![Pipeline](./Figures/pipeline.PNG)

## Dataset

- Link to dataset of images and the corresponding text: [here](https://drive.google.com/open?id=1Nbp9ZXLlWV3n8yRMwj2gjs_rE6qGZU01).
- We used 1000 images to generate character dataset that we used for training.

# Examples

## Line Segmentation

![Line](./Figures/line.png)

## Word Segmentation

![Word](./Figures/word.png)

## Character Segmentation

![Word](./Figures/char4.png)
![Word](./Figures/char3.png)
![Word](./Figures/char2.png)
![Word](./Figures/char1.png)

# Testing

**NOTE**: Make sure you have a folder with the truth output with same file names to compare it with the predicted text.

From within `src` folder run:

`python edit.py 'output/text' 'truth'`

![Test](./Figures/test.png)

# Performance

- Average accuracy: 95%.
- Average time per image: 16 seconds.

---

**NOTE**

We achieved these results when we used only the flatten image as feature.

---

# References

1. [An Efficient, Font Independent Word and Character Segmentation Algorithm for Printed Arabic Text](https://www.researchgate.net/publication/335562626_An_Efficient_Font_Independent_Word_and_Character_Segmentation_Algorithm_for_Printed_Arabic_Text).

2. [A Robust Line Segmentation Algorithm for Arabic Printed Text with Diacritics](https://www.researchgate.net/publication/317876029_A_Robust_Line_Segmentation_Algorithm_for_Arabic_Printed_Text_with_Diacritics).

3. [Arabic Character Segmentation Using Projection Based Approach with Profile's Amplitude Filter
   ](https://www.researchgate.net/publication/318205989_Arabic_Character_Segmentation_Using_Projection_Based_Approach_with_Profile's_Amplitude_Filter).
