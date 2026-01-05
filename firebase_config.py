import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate("firebase_admin_key.json")

firebase_admin.initialize_app(cred, {
    "storageBucket": "fine-cb713.appspot.com"
})

db = firestore.client()
bucket = storage.bucket()