import pyrebase

firebaseConfig = {
  "apiKey": "AIzaSyDJ3V8DPbjf0ipSUbCBOBd9vpGpOwAPZPk",
  "authDomain": "data-storage-f0a72.firebaseapp.com",
  "databaseURL": "https://data-storage-f0a72-default-rtdb.firebaseio.com",
  "projectId": "data-storage-f0a72",
  "storageBucket": "data-storage-f0a72.appspot.com",
  "messagingSenderId": "761040075298",
  "appId": "1:761040075298:web:1f3c8315b3b6d745a8e87b",
  "measurementId": "G-4G3J8DX0QL"
};
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
database = firebase.database()

notDrowsy = "Not Drowsy"
modDrowsy = "Moderately Drowsy"
xtmDrowsy = "Extremely Drowsy"

database.child("Status")
data = {"stat": notDrowsy}
database.set(data)
