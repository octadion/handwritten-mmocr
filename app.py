import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmocr.apis import TextDetInferencer, TextRecInferencer
from spellchecker import SpellChecker
import os

DBNETPP_IC15 = {
    "config": "models/detection/dbnet_trained.py",
    "weights": "models/detection/dbnet_trained.pth"
}

SVTR_SYNTH = {
    "config": "models/recognition/svtr_trained.py",
    "weights": "models/recognition/svtr_trained.pth"
}

def organize_points(rect):
    """
    Sort 4 vertices polygon into the same order
    [top-left, top-right, bottom-right, bottom-left]
    """
    points = sorted(list(rect), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0

    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    return np.array([
        points[index_1], points[index_2], points[index_3], points[index_4]
    ])

def extract_text_image(rect, img):
    """
    Given 4 vertices polygon, crop the image using that
    polygon, and transform the crop so that it is flat.
    """
    rect = organize_points(rect)
    tl, tr, br, bl = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def ocr_image(image_path, output_dir):
    img = mmcv.imread(image_path, channel_order="rgb")
    det_inferencer = TextDetInferencer(model=DBNETPP_IC15["config"], weights=DBNETPP_IC15["weights"])
    det_result = det_inferencer(img)
    polys_raw = det_result["predictions"][0]["polygons"]

    polys = []
    for poly in polys_raw:
        c = np.array(poly).astype(int).reshape((-1, 2))
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        polys.append(box)

    rec_inferencer = TextRecInferencer(model=SVTR_SYNTH["config"], weights=SVTR_SYNTH["weights"])

    spell = SpellChecker(language='en')

    texts = []
    for poly in polys:
        poly_arr = np.array(poly).reshape((-1, 2)).astype(np.float32)
        txt_img = extract_text_image(poly_arr, img)
        rec_result = rec_inferencer(txt_img)
        text = rec_result["predictions"][0]["text"]
        corrected_text = spell.correction(text)
        if corrected_text is not None:
            texts.append(corrected_text)

    texts.reverse()
    paragraph = ' '.join(texts)

    output_file = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
    with open(output_file, 'w') as f:
        f.write(paragraph)

def main(input_dir, output_dir):
    if os.path.isfile(input_dir):
        ocr_image(input_dir, output_dir)
    elif os.path.isdir(input_dir):
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(input_dir, filename)
                ocr_image(image_path, output_dir)
    else:
        print(f"Invalid input_dir: {input_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input images or path to an image file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output text files')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

