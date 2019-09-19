import cv2
import os
import numpy as np
import pyocr
import pyocr.builders
import pandas as pd
import re
import sys
from PIL import Image
from pdf2image import convert_from_path
import tempfile


def convertPDFtoJPG(convertImageFilePath):
    fileDetails = os.path.split(convertImageFilePath)
    filePath = fileDetails[0] + "/"
    fileName = os.path.splitext(fileDetails[1])[0]
    with tempfile.TemporaryDirectory() as path:
        images_from_path = convert_from_path(
            convertImageFilePath, output_folder=path,
            last_page=1, first_page=0)
    for page in images_from_path:
        page.save(os.path.join(filePath, fileName + ".jpg"), 'JPEG')
    return cv2.imread(filePath+fileName+".jpg")


def ocrInterface(imageFilePath, heightScaleFactor, ocrLanguage,
                 printResult=False, showImages=False, printOcrLines=False):
    """
    ### Description:
        Performs Optical Character Recognition (OCR) on an image.

    ### Arguments:
        imageFilePath {[String]} -- Filepath where from image is loaded.
        heightScaleFactor {[int]} -- Value that determine image height
        ocrLanguage {[String]} -- Language in which to perform OCR
        printResult {[boolean]} -- Should OCR result be printed?
        showImages {[boolean]} -- Should intermediate images be shown?
        printLines {[boolean]} -- Should each recognized line be printed?

    ### Returns:
        [pandas.DataFrame] -- DataFrame where each row represents a
        purchased item.
    """
    fileExtension = os.path.splitext(
        os.path.basename(imageFilePath))[1].lower()

    if fileExtension == ".pdf":
        image = convertPDFtoJPG(imageFilePath)
        isPDF = True
    else:
        image = cv2.imread(imageFilePath)
        isPDF = False

    rescalingFactor = heightScaleFactor/image.shape[0] * 100
    if not isPDF:
        image = addPaddingToImage(image, 100)

    canImage = applyCannySquareEdgeDetectionOnImage(
        image, rescalingFactor)
    ROI = findLargestSquareOnCannyDetectedImage(canImage)
    warpImage = getBirdView(image, ROI, rescalingFactor)
    cleanImage = cleanImageForOCR(warpImage)
    ocrResult = ocrEngine(cleanImage, ocrLanguage, printOcrLines)

    if showImages:
        cv2.drawContours(canImage, [ROI], -1, (255, 255, 255), 3)
        cv2.imshow("Padded", image)
        cv2.imshow("Canny Edge", canImage)
        cv2.imshow("Warped", warpImage)
        cv2.imshow("Clean", cleanImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if printResult:
        print(ocrResult)

    return ocrResult


def downScaleImage(srcImage, percent):
    """
    ### Description:
        Scale srcImage according to percent

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()
        percent {[int]} -- Rescaling factor (for faster computation)

    ### Returns:
        [cv2.Image] -- Rescaled image
    """
    print("srcImage - Height - ",
          srcImage.shape[0], ", width - ", srcImage.shape[1])
    destImage = cv2.resize(srcImage, None, fx=percent/100, fy=percent/100)
    print("destImage - Height - ",
          destImage.shape[0], ", width - ", destImage.shape[1])
    return destImage


def applyCannySquareEdgeDetectionOnImage(srcImage, percent):
    """
    ### Description:
        Applies Canny edge detection algorithm on srcImage.

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()
        percent {[int]} -- Rescaling factor (for faster computation)

    ### Returns:
        [cv2.Image] -- Image with edges detected by Canny
    """
    destImage = downScaleImage(srcImage, percent)
    # Convert to Grayscale
    grayImage = cv2.cvtColor(destImage, cv2.COLOR_BGR2GRAY)

    # Add Gaussian Blur

    destImage = cv2.GaussianBlur(
        grayImage, (9, 9), 0, 0, borderType=cv2.BORDER_DEFAULT)
    destImage = cv2.bilateralFilter(destImage, 11, 150, 150)
    destImage = cv2.threshold(destImage, 127, 255, cv2.THRESH_BINARY)[1]
    # Clean Image for better Detection
    kernel = np.ones((5, 5), np.uint8)
    destImage = cv2.erode(grayImage, kernel, iterations=1)
    destImage = cv2.dilate(destImage, kernel, iterations=1)
    # Apply Canny Edge Detection
    destImage = cv2.Canny(destImage, 75, 200, L2gradient=True)
    return destImage


def findLargestSquareOnCannyDetectedImage(edgedImage):
    """
    ### Description:
        Finds largest contour on edged image, which have been processed
        by an appropriate edging algorithm, e.g. Canny.

    ### Arguments:
        edgedImage {[cv2.Image]} -- Preprocessed image by e.g. cv2.Canny()

    ### Returns:
        [np.array] -- Region Of Interest from edged image
    """
    foundContourImage = edgedImage.copy()
    contours, hierarchy = cv2.findContours(
        foundContourImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest Contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    seqFound = None
    maxWidth = 0
    maxHeight = 0

    for contour in contours:
        width = cv2.boundingRect(contour)[2]
        height = cv2.boundingRect(contour)[3]
        if width >= maxWidth & height >= maxHeight:
            maxWidth = width
            maxHeight = height
            seqFound = contour

    # Outline polygon of largest Contour
    epsilon = cv2.arcLength(seqFound, True) * 0.05
    result = cv2.approxPolyDP(seqFound, epsilon, True)
    return result


def orderLargestSquare(ROI):
    """
    ### Description:
        Initialzie a list of coordinates that will be ordered
        such that the first entry in the list is the top-left,
        the second entry is the top-right, the third is the
        bottom-right, and the fourth is the bottom-left

    ### Arguments:
        ROI {[np.array]} -- ROI obtained through cv2.approxPolyDP()

    ### Returns:
        [np.array[4, 2]] -- Rearranged ROI
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = ROI.reshape(ROI.shape[0], ROI.shape[2])

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def getBirdView(srcImage, ROI, percent):
    """
    ### Description:
        Rearrange image to fit region of interest (ROI) and return
        an image warped to birds eye view.

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()
        ROI {[np.array]} -- ROI obtained through cv2.approxPolyDP()
        percent {[int]} -- Rescaling factor (for faster computation)

    ### Returns:
        [cv2.Image] -- ROI of srcImage as seen from birds eye view
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = orderLargestSquare(ROI)
    rect = (rect*100)/percent
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(srcImage, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def cleanImageForOCR(srcImage):
    """
    ### Description
        Cleans an Image by applying grayscale, medianblur,
        and otsu threshold prior to OCR

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()

    ### Returns:
        destImage [cv2.Image] -- A clean image
    """
    destImage = srcImage.copy()
    destImage = cv2.cvtColor(destImage, cv2.COLOR_BGR2GRAY)
    destImage = cv2.medianBlur(destImage, 3)
    destImage = cv2.threshold(destImage, 0, 255, cv2.THRESH_OTSU)[1]
    return destImage


def addPaddingToImage(srcImage, paddingSize):
    """
    ### Description:
        Adds padding around an image corresponding to the average
        RGB values of both left and right sides

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()
        paddingSize {[int]} -- Determines size of padding in Pixels

    ### Returns:
        [cv2.Image] -- srcImage with padding on all sides
    """
    destImage = srcImage.copy()
    row, col = destImage.shape[:2]
    right = destImage[0:row, col-2:col]
    left = destImage[0:row, 0:2]
    br, gr, rr, _ = cv2.mean(right)
    bl, gl, rl, _ = cv2.mean(left)
    b, g, r = (np.mean((br, bl)), np.mean((gr, gl)), np.mean((rr, rl)))

    if (b > 200) & (g > 200) & (r > 200):
        b, g, r = (0, 0, 0)
    destImage = cv2.copyMakeBorder(
        src=destImage,
        top=paddingSize,
        bottom=paddingSize,
        left=paddingSize,
        right=paddingSize,
        borderType=cv2.BORDER_CONSTANT,
        value=[b, g, r]
    )
    return destImage


def pullPurchaseInfo(string):
    """
    ### Description:
        Extract purchase information from string basex from Regex.

    ### Arguments:
        string {[String]} -- String with potential purchase information.

    ### Returns:
        [Dict] -- Dictionary with item description and price
    """
    s = re.search(r'(\S+\ \S+|)(\S+\ )(\d+(,|\.)\d\d)', string)
    if (s is None):
        return None

    item = re.search(r'(\S+\ \S+|)(\S+\ )', string).group()
    item = re.sub(' ', '', item)
    price = re.search(r'(\d+(,|\.)\d\d)', string).group()
    price = re.sub(r'\.', '', price)
    price = re.sub(r',', '.', price)
    Dict = eval("{" +
                "'desc' : '" + item + "'," +
                "'price' : " + price + '}')
    return Dict


def ocrEngine(srcImage, language, printOutput):
    """
    ### Description:
        Perform OCR on srcImage in a given language.

    ### Arguments:
        srcImage {[cv2.Image]} -- Image loaded by cv2.imread()
        language {[String]} -- Indication of the language to be used for OCR
        printOutput {[boolean]} -- Should output be printed?

    ### Returns:
        [pandas.DataFrame] -- DataFrame with item description and price
    """

    destImage = srcImage.copy()
    destImage = cv2.cvtColor(destImage, cv2.COLOR_BGR2RGB)
    destImage = Image.fromarray(destImage)

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    print("Will use lang '%s'" % (language))

    # list of line objects. For each line object:
    #   line.word_boxes is a list of word boxes (individual words in the line)
    #   line.content is the whole text of the line
    #   line.position is the position of the whole line on the page (in pixels)
    line_and_word_boxes = tool.image_to_string(
        destImage, lang=language,
        builder=pyocr.builders.LineBoxBuilder()
    )
    df = pd.DataFrame(columns=('desc', 'price'))
    for line in line_and_word_boxes:
        if printOutput:
            print(line.content)

        txt = pullPurchaseInfo(line.content)
        if txt is None:
            continue
        else:
            df = df.append(txt, ignore_index=True)

    return df


# Set Working Directory
wdpath = '/Users/jenspedermeldgaard/Google Drive/CS/PythonProjects/OCR/'

# Set IO Parameters
inputDirectory = wdpath + 'images/'
outputDirectory = wdpath + 'out/'
allowedExtensions = tuple(('.jpg', '.jpeg', '.png', ".pdf"))

# Pull File Details
files = os.listdir(inputDirectory)
files = [str.lower(f) for f in files]
files = [f for f in files if f.endswith(allowedExtensions)]

for f in files:
    print("\n\nInitiating OCR on file " + f + "...")
    inPath = inputDirectory + f
    result = ocrInterface(inPath, 800, "dan", False, False, False)
    outPath = outputDirectory + \
        os.path.splitext(os.path.basename(f))[0] + ".csv"
    result.to_csv(outPath, index=None, header=True)
