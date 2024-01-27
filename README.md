# Handwritten MMOCR

This project is an implementation of Optical Character Recognition (OCR) using MMOCR from training until inferencing, it's also testing various models such as PaddleOCR, TesseractOCR, EasyOCR to detect and recognize text from images. The recognized text is then evaluated for correctness using Character Error Rate (CER) and Word Error Rate (WER) metrics calculated by PyWER, JiWER, and FastWER libraries.

The code is designed to perform Optical Character Recognition (OCR) on images. It utilizes two models: DBNet for detecting text areas in the images, and SVTR for recognizing the actual text within these detected areas. Once the text is recognized, it’s corrected for any spelling errors using a spell checker. The corrected text is then saved to a text file. This entire process can be applied to a single image or multiple images in a directory, depending on the input provided to the script.

## Workflow

The workflow of this project is as follows:

1. An image is input to the OCR models.
2. The models detect areas in the image that contain text.
3. The detected text areas are recognized as actual text.
4. Save the recognized text as TXT file.

## Results

The following table shows the CER and WER scores for each model using each metric library:

| Model | Metric Library | CER | WER |
|-------|----------------|-----|-----|
| PaddleOCR | JiWER | 0.2113 | 0.7428 |
| TesseractOCR | JiWER | 0.493 | 1.042 |
| EasyOCR | JiWER | 0.4 | 1.014 |
| MMOCR Base | JiWER |  0.502 | 1.028 |
| MMOCR Trained | JiWER | 0.315 | 0.585 |
