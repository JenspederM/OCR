# %%
from PIL import Image
import sys
import pyocr
import pyocr.builders
import os
import pandas as pd
import re

# Set Working Directory
wdpath = '/Users/jenspedermeldgaard/Google Drive/CS/PythonProjects/OCR/'

# Set IO Parameters
inputDirectory = wdpath + 'images/'
outputDirectory = wdpath + 'out/'
allowedExtensions = tuple(('.jpg', '.jpeg', '.png'))
acceptedLineStarts = tuple((
    'total', 'dankort', 'atbetale', 'kortbetaling',
    'ialt'))

# Pull File Details
files = os.listdir(inputDirectory)
files = [str.lower(f) for f in files]
files = [f for f in files if f.endswith(allowedExtensions)]
impath = inputDirectory + files[3]

img = Image.open(impath)
