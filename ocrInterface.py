from waffleOCR.scannerFunctions import ocrInterface
from waffleOCR.firestore import connectToFirestoreStorage
from waffleOCR.firestore import grapBankStaments
from waffleOCR.firestore import detect_text_uri

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/jenspedermeldgaard/waffle-kasserer-creds.json'
# Mangler "Enable Billing" --> Så skulle det gerne virke
bucket = connectToFirestoreStorage()
blobs = bucket.list_blobs()
blobpaths = [blob.name for blob in blobs]
storage_bucket = "gs://waffle-kasserer.appspot.com/"


detect_text_uri(storage_bucket + blobpaths[0])


# print(blobpaths)


# bankStatements = grapBankStaments()
# print(bankStatements)

# # region Work Space
# # Set Working Directory
# wdpath = '/Users/jenspedermeldgaard/Google Drive/CS/PythonProjects/OCR/'

# # Set IO Parameters
# inputDirectory = wdpath + 'images/'
# outputDirectory = wdpath + 'out/'
# allowedExtensions = tuple(('.jpg', '.jpeg', '.png', ".pdf"))


# # Pull File Details
# files = os.listdir(inputDirectory)
# files = [str.lower(f) for f in files]
# files = [f for f in files if f.endswith(allowedExtensions)]

# for f in files:
#     print("\n\nInitiating OCR on file " + f + "...")
#     inPath = inputDirectory + f
#     result = ocrInterface(inPath, 800, "dan", False, False, False)
#     outPath = outputDirectory + \
#         os.path.splitext(os.path.basename(f))[0] + ".csv"
#     result.to_csv(outPath, index=None, header=True)
# # endregion
