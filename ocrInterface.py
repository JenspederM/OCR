from waffleOCR.scannerFunctions import ocrInterface
from waffleOCR.firestore import grapBankStaments
import os


bankStatements = grapBankStaments()
print(bankStatements)

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
