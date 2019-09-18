# %%
from PIL import Image
import sys
import pyocr
import pyocr.builders
import os
import pandas as pd
import re

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = 'dan'
print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.


# %%
def pullPurchaseInfo(line):
    s = re.search(r'(\S+\ \S+|)(\S+\ )(\d+(,|\.)\d\d)', line)
    if (s is None):
        return None

    item = re.search(r'(\S+\ \S+|)(\S+\ )', line).group()
    item = re.sub(' ', '', item)
    price = re.search(r'(\d+(,|\.)\d\d)', line).group()
    price = re.sub(r'\.', '', price)
    price = re.sub(r',', '.', price)
    Dict = eval("{" +
                "'desc' : '" + item + "'," +
                "'price' : " + price + '}')
    return Dict


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

# %%
line_and_word_boxes = tool.image_to_string(
    img, lang="dan",
    builder=pyocr.builders.LineBoxBuilder()
)
# list of line objects. For each line object:
#   line.word_boxes is a list of word boxes (the individual words in the line)
#   line.content is the whole text of the line
#   line.position is the position of the whole line on the page (in pixels)
#

df = pd.DataFrame(columns=('desc', 'price'))
for line in line_and_word_boxes:
    print(line.content)
    txt = pullPurchaseInfo(line.content)
    if txt is None:
        continue
    else:
        df = df.append(txt, ignore_index=True)

print(df)


# %%
df.loc[0:]
