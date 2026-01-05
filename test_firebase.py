from firebase_config import db

db.collection("test_connection").add({
    "message": "Python successfully connected to Firebase"
})

print("✅ Firebase connection successful")