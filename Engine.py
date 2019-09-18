from __future__ import print_function
import re
import os
import pytesseract
import cv2
import numpy as np
from wand.image import Image


def pullPurchaseInfo(line):
    s = re.search(r'(\S+\ \S+|)(\S+\ )(\d+(,|\.)\d\d)', line)
    if (s is None):
        return None

    item = re.search(r'(\S+\ \S+|)(\S+\ )', line).group()
    item = re.sub(' ', '', item)
    price = re.search(r'(\d+(,|\.)\d\d)', line).group()
    price = re.sub(r'\.', '', price)
    price = re.sub(r',', '.', price)
    return item + " , " + price


def alignReceipt(image):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    print("[INFO] angle: {:.3f}".format(angle))
    return rotated


def ocrCore(image, config=''):
    text = pytesseract.image_to_string(image, config=config)
    return text


# Define config parameters.
# '-l eng'  for using the English language --> -l = Language
# '--oem 2' for using LSTM OCR + Legacy Engine --> -oem = OCR Engine Mode
# '--psm 1' for using ????? Can't remember --> -psm = Page Segmentation Mode
ocrConfig = ('-l eng --oem 2 --psm 1')

# Set Working Directory
wdpath = '/Users/jenspedermeldgaard/Google Drive/CS/PythonProjects/OCR/'

# Set IO Parameters
inputDirectory = wdpath + 'images/'
outputDirectory = wdpath + 'out/'
allowedExtensions = tuple(('.jpg', '.jpeg', '.png'))
acceptedLineStarts = tuple(('total', 'dankort', 'atbetale', 'kortbetaling'))

# Pull File Details
files = os.listdir(inputDirectory)
files = [str.lower(f) for f in files]
files = [f for f in files if f.endswith(allowedExtensions)]

# Loop Through Files and Pull Purchase Information
for f in files:
    print('OCR on ' + f)

    # Load and Preprocess Image
    impath = inputDirectory + f
    image = Image(filename=impath)

    with image.clone() as liquid:
        liquid.liquid_rescale(liquid.size[0]*2, liquid.size[1]*2)
        liquid.adaptive_sharpen(radius=8, sigma=4)
        liquid.save(filename=outputDirectory + 'tmp/' + f)

    # Perform OCR
    testpath = outputDirectory + f
    testimage = cv2.imread(testpath)
    testimage = alignReceipt(testimage)
    ocr_result = ocrCore(testimage, config=ocrConfig)

    # Write Order Lines to CSV
    csv_file = open(outputDirectory + "csv/" + f.split('.')[-2] + ".csv", "w")
    cv2.imwrite(outputDirectory + 'images/' +
                f.split('.')[-2] + '.png', testimage)

    for cur_line in ocr_result.split("\n"):
        result = pullPurchaseInfo(cur_line)

        if result is None:
            continue

        if result.lower().startswith(acceptedLineStarts):
            csv_file.write(result + '\n')
        else:
            continue

    csv_file.close()
