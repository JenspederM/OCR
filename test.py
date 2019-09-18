# %%
import cv2
import os
import numpy as np

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

image = cv2.imread(inputDirectory + files[1])

# %%


def downScaleImage(srcImage, percent):
    print("srcImage - Height - ",
          srcImage.shape[0], ", width - ", srcImage.shape[1])
    destImage = cv2.resize(srcImage, None, fx=percent/100, fy=percent/100)
    print("destImage - Height - ",
          destImage.shape[0], ", width - ", destImage.shape[1])
    return destImage


def applyCannySquareEdgeDetectionOnImage(srcImage, percent):
    destImage = downScaleImage(srcImage, percent)
    # Convert to Grayscale
    grayImage = cv2.cvtColor(destImage, cv2.COLOR_BGR2GRAY)
    # Add Gaussian Blur
    destImage = cv2.GaussianBlur(
        grayImage, (5, 5), 0, 0, borderType=cv2.BORDER_DEFAULT)
    # Clean Image for better Detection
    kernel = np.ones((5, 5), np.uint8)
    destImage = cv2.erode(destImage, kernel, iterations=1)
    destImage = cv2.dilate(destImage, kernel, iterations=1)
    # Apply Canny Edge Detection
    destImage = cv2.Canny(destImage, 75, 200)
    return destImage


def findLargestSquareOnCannyDetectedImage(cannyEdgeDetectedImage):
    foundContourImage = cannyEdgeDetectedImage.copy()
    contours, hierarchy = cv2.findContours(
        foundContourImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest Contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    seqFound = contours[0]

    # Outline polygon of largest Contour
    epsilon = cv2.arcLength(seqFound, True) * 0.02
    result = cv2.approxPolyDP(seqFound, epsilon, True)
    return result


def orderLargestSquare(largeSquare):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    pts = largeSquare.reshape(largeSquare.shape[0], largeSquare.shape[2])

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


def getBirdView(image, pts, percent):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = orderLargestSquare(pts)
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# %%
canImage = applyCannySquareEdgeDetectionOnImage(image, 30)

# %%
largeSquare = findLargestSquareOnCannyDetectedImage(canImage)

# %%


def addPaddingToImage(srcImage, paddingSize):
    destImage = srcImage.copy()
    row, col = destImage.shape[:2]
    right = destImage[0:row, col-2:col]
    left = destImage[0:row, 0:2]
    mean = np.mean((cv2.mean(right), cv2.mean(left)))
    destImage = cv2.copyMakeBorder(
        src=destImage,
        top=paddingSize,
        bottom=paddingSize,
        left=paddingSize,
        right=paddingSize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    return destImage

# %%


# %%
int(1024/1480*100)

# %%
