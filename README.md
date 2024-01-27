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
| PaddleOCR | PyWER | 92.31 | 109.70 |
| PaddleOCR | JiWER | 0.2113 | 0.7428 |
| PaddleOCR | FastWER | 21.59 | 74.28 |
| PaddleOCR | Manual Calc | 0.988 | 1.185 |
| TesseractOCR | PyWER | 93.84 | 111.55 |
| TesseractOCR | JiWER | 0.493 | 1.042 |
| TesseractOCR | FastWER |  51.59 | 104.28 |
| TesseractOCR | Manual Calc |  1.454 | 1.885 |
| EasyOCR | PyWER | 94.87 | 112.74 |
| EasyOCR | JiWER | 0.4 | 1.014 |
| EasyOCR | FastWER | 40.45 | 101.4 |
| EasyOCR | Manual Calc | 1.597 | 1.957 |
| MMOCR Base | PyWER | 89.90 | 107.16 |
| MMOCR Base | JiWER |  0.502 | 1.028 |
| MMOCR Base | FastWER | 50.2 | 102.85 |
| MMOCR Base | Manual Calc | 1.522 | 1.8 |
| MMOCR Trained | PyWER | 90.53 | 107.69 |
| MMOCR Trained | JiWER | 0.315 | 0.585 |
| MMOCR Trained | FastWER | 31.5 | 58.5 |
| MMOCR Trained | Manual Calc | 1.084 | 0.9285 |
