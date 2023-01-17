import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://face-attendance-realtime-default-rtdb.firebaseio.com/"
})

ref = db.reference("Students")

data = {
    "123456":
    {
        "name" : "Arnab Kumar Roy",
        "major" : "Computer Science & Engineering",
        "starting_year" : 2020,
        "total_attendance" : 0,
        "standing" : "G",
        "year" : 3,
        "last_attendance_time" : "17-01-2023 21:35:00"
    }
}

for key, value in data.items():
    ref.child(key).set(value)
