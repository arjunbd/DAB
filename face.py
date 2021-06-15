#standalone face detection
import pymongo
#from io import BytesIO
#from PIL import Image
import ssl
import numpy as np
import face_recognition
import cv2
import os
from gtts import gTTS
import vlc

language = 'en'

camera = cv2.VideoCapture(0)
for c in range(30):
    return_value, capture = camera.read()
del(camera)
client = pymongo.MongoClient("mongodb+srv://basil:project%40123@dab.iz1ar.mongodb.net/DAB?retryWrites=true&w=majority", ssl_cert_reqs=ssl.CERT_NONE)
# client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["DAB"]
known_identity = db["identity"].find({})

#if known_identity.count()==0:

cv2.imwrite(filename='saved_img.jpg', img=capture)

capture = face_recognition.load_image_file("saved_img.jpg")

capture =face_recognition.face_encodings(capture)
if capture ==[]:
    print("not found")
    myobj = gTTS(text="person not found", lang=language, slow=False)
    myobj.save("welcome.mp3")
    media = vlc.MediaPlayer("welcome.mp3")
    media.play()
for unknown in capture:
    for known in known_identity:
        known_image = np.asarray(known["image"])
        #known_image = face_recognition.face_encodings(known_image)[0]
        results = face_recognition.compare_faces([known_image], unknown)
        if results[0] == True:
            myobj = gTTS(text=known["name"], lang=language, slow=False)
            myobj.save("welcome.mp3")
            media = vlc.MediaPlayer("welcome.mp3")
            media.play()
            print(known["name"])
        else:
            print("not found")
            myobj = gTTS(text="person not found", lang=language, slow=False)
            myobj.save("welcome.mp3")
            media = vlc.MediaPlayer("welcome.mp3")
            media.play()


