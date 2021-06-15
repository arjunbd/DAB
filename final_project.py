# driver package
import RPi.GPIO as GPIO
import time

# face_detection package
import pymongo
import ssl
import face_recognition
import cv2
import os
from gtts import gTTS
import vlc

#object_detection package
import argparse
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

#ocr package
from PIL import Image
import pytesseract


def face_detection():
    print("Starting Face Detection")
    language = 'en'

    camera = cv2.VideoCapture(0)
    for c in range(30):
        return_value, capture = camera.read()
    del (camera)
    client = pymongo.MongoClient(
        "mongodb+srv://basil:project%40123@dab.iz1ar.mongodb.net/DAB?retryWrites=true&w=majority",
        ssl_cert_reqs=ssl.CERT_NONE)
    # client = pymongo.MongoClient("mongodb://localhost:27017/")

    db = client["DAB"]
    known_identity = db["identity"].find({})

    # if known_identity.count()==0:

    cv2.imwrite(filename='saved_img.jpg', img=capture)

    capture = face_recognition.load_image_file("saved_img.jpg")

    capture = face_recognition.face_encodings(capture)
    if capture == []:
        print("not found")
        myobj = gTTS(text="person not found", lang=language, slow=False)
        myobj.save("sound.mp3")
        media = vlc.MediaPlayer("sound.mp3")
        media.play()
    for unknown in capture:
        for known in known_identity:
            known_image = np.asarray(known["image"])
            # known_image = face_recognition.face_encodings(known_image)[0]
            results = face_recognition.compare_faces([known_image], unknown)
            if results[0] == True:
                myobj = gTTS(text=known["name"], lang=language, slow=False)
                myobj.save("sound.mp3")
                media = vlc.MediaPlayer("sound.mp3")
                media.play()
                print(known["name"])
            else:
                print("not found")
                myobj = gTTS(text="person not found", lang=language, slow=False)
                myobj.save("sound.mp3")
                media = vlc.MediaPlayer("sound.mp3")
                media.play()
    #face module


def object_detection():
    print("Starting object Detection")

    class VideoStream:

        def __init__(self, resolution=(640, 480), framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
            # Start the thread that reads frames from the video stream
            Thread(target=self.update, args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
            # Return the most recent frame
            return self.frame

        def stop(self):
            # Indicate that the camera and thread should be stopped
            self.stopped = True



    MODEL_NAME = "object_detection"
    GRAPH_NAME = "detect.tflite"
    LABELMAP_NAME = "labelmap.txt"
    min_conf_threshold = float(0.5)
    
    imW, imH = 1280, 720
    
    from tflite_runtime.interpreter import Interpreter

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del (labels[0])

    # Load the Tensorflow Lite model.  
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    final = []
    for q in range(3):

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                if object_name not in final:
                    final.append(object_name)

        if final==[]:
            final.append("no object")

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    for o in final:
        myobj = gTTS(text=o, lang="en", slow=False)
        myobj.save("sound.mp3")
        media = vlc.MediaPlayer("sound.mp3")
        media.play()
        time.sleep(1)
    #object module


def character_detection():
    
    print("Starting Optical Character Recognition")
    
    camera = cv2.VideoCapture(0)
    for c in range(20):
        return_value, img = camera.read()
    del (camera)
    cv2.imwrite(filename='test.jpg', img=img)
    # img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    denoise = cv2.fastNlMeansDenoisingColored(thresh1, None, 10, 10, 7, 21)
    #cv2.imwrite('gt.png', denoise)
    blur = cv2.GaussianBlur(denoise, (5, 5), 0)
    grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('hulk.png', grey)
    original = pytesseract.image_to_string(grey, config=' ')
    original = original.replace("\n", " ")
    print(original)
    try:
        sound = gTTS(original, lang="en")
        sound.save("sound.mp3")
    except:
        original="no text"
        sound = gTTS(original, lang="en")
        sound.save("sound.mp3")
        
    media = vlc.MediaPlayer("sound.mp3")
    media.play()
    #ocr module

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low(off)
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True: # Run forever

    if GPIO.input(10) == GPIO.HIGH:
       face_detection()
       time.sleep(1)

    elif GPIO.input(11) == GPIO.HIGH:
        object_detection()
        time.sleep(1)

    elif GPIO.input(12) == GPIO.HIGH:
        character_detection()
        time.sleep(1)

GPIO.cleanup() # Clean up
