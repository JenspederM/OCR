import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
import numpy as np


def connectToFireStoreDataBase():
    cred = credentials.Certificate(
        '/Users/jenspedermeldgaard/waffle-kasserer-creds.json')

    firebase_admin.initialize_app(cred, {
        'projectId': 'waffle-kasserer',
    })

    return firestore.client()


def grapBankStaments():
    db = connectToFireStoreDataBase()

    docs = db.collection(u'Foreninger').document(u'AACP').collection(
        u'Budget').document(u'2019').collection(u'BankStatements')\
        .where(u'amount', u'<', 0).where(u'appendixSet', u'==', True).stream()

    bankStatements = pd.DataFrame(columns=('description', 'amount'))

    for doc in docs:
        fsDict = doc.to_dict()
        fsDict = {key: fsDict[key] for key in ('amount', 'description')}
        bankStatements = bankStatements.append(fsDict, ignore_index=True)

    # bankStatements = bankStatements[bankStatements['amount'] < 0]
    bankStatements['amount'] = np.absolute(bankStatements['amount'])
    return bankStatements
