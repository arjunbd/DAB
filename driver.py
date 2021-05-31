import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import time
#import packages

def face_detection():
    print("face")
    #face module

def object_detection():
    print("object")
    #object module

def character_detection():
    print("OCR")
    #ocr module

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


#GPIO.add_event_detect(10,GPIO.RISING,callback=face_detection)
#GPIO.add_event_detect(11,GPIO.RISING,callback=object_detection)

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
