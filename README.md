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

## Prerequisites

Ensure you have the following dependencies installed:

- PyTorch
- torchvision
- cuDNN
- OpenMIM
- MMOCR
- PySpellChecker

## Dataset Creation and Labelling

This project requires a handwritten dataset. You can use the dataset example in [`handwritten-mmocr/dataset/`](./dataset/). Follow these steps if you want create and label your dataset:

1. Collect handwritten samples for your dataset.
2. Install and set up Label Studio.
3. Import your collected samples into Label Studio.
4. Label the samples according to your project requirements.

Ensure the dataset is properly labeled and saved in a format compatible with the OCR models used in this project.

## Installation

To install this project, follow these steps:

1. Clone this repository.
2. Download the models from the provided link ([Text Detection Model](https://drive.google.com/file/d/1-7JOWZhkmGrpqGH6hOCXBtYIMR25eCwn/view?usp=sharing) | [Text Recognition Model](https://drive.google.com/file/d/1-3Mfdh6f5f4V1EdKDCicr_5X_jQzFfGW/view?usp=sharing)).
3. Place the downloaded models into the appropriate folders [`handwritten-mmocr/models/`](./models/).
4. Update the model paths in the `app.py` file to match the locations of your downloaded models.
5. Open your terminal and navigate to the project directory.
6. Run the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
```
Note: If you encounter an error when using mmdet, you can install it using OpenMIM with the following command: `mim install mmdet`.

## Usage

You can run this project with the following command:

```bash
python app.py --input_dir <input_dir_or_image_path> --output_dir <output_dir>
```
Where:

<input_dir_or_image_path> is a directory containing images or a path to a specific image file.
<output_dir> is the directory where the OCR results will be saved.

## Training

To train the models, follow these steps:

1. Open the `training.ipynb` file.
2. Run the cells and follow the instructions provided in the notebook.

Before training, make sure to modify the configurations in the `config/texdet/dbnet` and `config/textrecog/svtr` files, as well as the corresponding base files. 

For the text detection model (DBNet), the following configurations should be updated:
- `root_data`
- `num of iteration`
- `val cycle`
- `tensorboard visualizer`
- `save last checkpoint`

For the text recognition model (SVTR), the following configurations should be updated:
- `root_data`
- `num of iteration`
- `tensorboard visualizer`
- `save last checkpoint`
- `validation evaluator`
- `train/test dataset list`
- `update pretrained model url`

## Related Projects

You might also be interested in the following related project:

- [Semantic Entity Recognition of Handwritten Images using LayoutLMV3](https://github.com/octadion/handwritten-layoutlmv3): This project focuses on extracting information from images and save it in json key-value pair format. 
