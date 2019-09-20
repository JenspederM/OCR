import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import pandas as pd
import numpy as np


def explicit():
    from google.cloud import storage

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        '/Users/jenspedermeldgaard/waffle-kasserer-creds.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)


def detect_text_uri(uri):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = uri

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))


def connectToFirestoreStorage():
    """
    ### Description:
        Connects to Firestore Storage under project waffle-kasserer

    ### Returns:
        [firestore.client] -- Storage bucket connected to Firestore
    """
    cred = credentials.Certificate(
        '/Users/jenspedermeldgaard/waffle-kasserer-creds.json')

    firebase_admin.initialize_app(cred, {
        'storageBucket': 'waffle-kasserer.appspot.com'
    })

    return storage.bucket()


def connectToFirestoreDatabase():
    """
    ### Description:
        Connects to Firestore Database under project waffle-kasserer

    ### Returns:
        [firestore.client] -- Client connected to Firestore database
    """
    cred = credentials.Certificate(
        '/Users/jenspedermeldgaard/waffle-kasserer-creds.json')

    firebase_admin.initialize_app(cred, {
        'projectId': 'waffle-kasserer',
    })

    return firestore.client()


def grapBankStaments():
    """
    ### Description:
        Grab all negative bank statements, which haven't yet been
        paired with an appendix from Firestore database.

    ### Returns:
        [pandas.DataFrame] -- DataFrame with item description and price
    """
    db = connectToFirestoreDatabase()

    docs = db.collection(u'Foreninger').document(u'AACP').collection(
        u'Budget').document(u'2019').collection(u'BankStatements')\
        .where(u'amount', u'<', 0).where(u'appendixSet', u'==', True).stream()

    bankStatements = pd.DataFrame(columns=('description', 'amount'))

    for doc in docs:
        fsDict = doc.to_dict()
        fsDict = {key: fsDict[key] for key in ('amount', 'description')}
        bankStatements = bankStatements.append(fsDict, ignore_index=True)

    bankStatements['amount'] = np.absolute(bankStatements['amount'])
    return bankStatements
