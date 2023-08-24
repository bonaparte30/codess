import serial
import time
import string
import pynmea2
import threading
from firebase.firebase import FirebaseApplication, FirebaseAuthentication

while True :
    if __name__ == '__main__':
        port="/dev/ttyAMA0"
        ser=serial.Serial(port, baudrate=9600, timeout=1)
        dataout = pynmea2.NMEAStreamReader()
        newdata=ser.readline().decode('unicode_escape')

        if newdata[0:6] == "$GPRMC":
            newmsg=pynmea2.parse(newdata)
            lat=newmsg.latitude
            lng=newmsg.longitude
            gps = "Latitude=" + str(lat) + " and Longitude=" + str(lng)
            
            newlat=lat
            newlng=lng
            #print(newlat)
            print(gps)
            
            SECRET = 'zJMoo1f9JaxL0OhyC7oGyjI5F8qj7qJi8gdoEUaI'
            DSN = 'https://data-storage-f0a72-default-rtdb.firebaseio.com'
            EMAIL = 'https://data-storage-f0a72-default-rtdb.firebaseio.com/'
            authentication = FirebaseAuthentication(SECRET, EMAIL, True, True,)
            firebase = FirebaseApplication(DSN, authentication)
            
    result = firebase.patch('/gps',
        {
            'latitude': newlat,
            'longitude': newlng
        })
    time.sleep(2)
    print(gps)
