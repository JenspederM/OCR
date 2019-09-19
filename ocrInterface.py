from pyimagesearch.scannerFunctions import ocrInterface
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


# region Firestore Credentials
# Use a service account
cred = credentials.Certificate(
    '/Users/jenspedermeldgaard/waffle-kasserer-creds.json')

firebase_admin.initialize_app(cred, {
    'projectId': 'waffle-kasserer',
})

db = firestore.client()

docs = db.collection(u'Foreninger').document(u'AACP').collection(
    u'Budget').document(u'2019').collection(u'BankStatements').stream()

for doc in docs:
    fsId = doc.id
    fsDict = doc.to_dict()
    print(u'{} => {}'.format(fsId, fsDict))

# endregion

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
