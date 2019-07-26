# IDEX
Vietnamese ID card OCR system from raw image

## Introduction
This is a program we developed as our side project to detect and extract information from latest type of Vietnamese ID card, we don't write this to work for other types of cards.

## Dependencies
    1. tensorflow 1.14.0
    2. opencv3
    3. numpy
    4. editdistance
    5. scipy
    6. tesseract
    7. pytesseract
    
## Installation
    1. Install the requirements
    2. Get the tessdata folder from this link: https://github.com/tesseract-ocr/tessdata and put this in your Tesseract-OCR folder
    3. Download saved_models from: https://drive.google.com/file/d/1RiEI_j8DQzJEh2U-82IVoJ7oqW7K7X-g/view?usp=sharing and extract it to IDEX
    4. Run run.py
    

## Requirements
    Python3 (3.6)
    
## How to run
    cd IDEX
    python run.py --input path/to/image.jpg --output path/to/output (you can skip the output part if you only want to see the result)
    
    
## Limitations
    1. Can't run for images that contain more than 1 card.
    2. Can't run for images whose background color and ID card color have low contrast (since we are using classical computer vision)
    3. Can't run for images with bad lighting (ID card edges covered in shadow,...)
    4. Doesn't 100% give the right anwer (obviously, duh)
    5. The saved_model only works for latest type of Vietnamese ID card image, and only detect names, id number and DoB, but you can always train a new model and improve the program.
    
