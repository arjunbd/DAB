#standalone OCR
from PIL import Image
import pytesseract
import cv2
from subprocess import call
import numpy as np
#from gtts import gtts

audio="speech.mp3"
language='en'

#add code here to capture image.Use either
#cmd='sudo fswebcam -S 20 --no-banner /home/pi/Desktop/Capture.jpeg'
#call([cmd],shell=True)
#or
##
img=cv2.imread("test.jpeg")
#img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
denoise = cv2.fastNlMeansDenoisingColored(thresh1,None,10,10,7,21)
cv2.imwrite('gt.png',denoise)
blur = cv2.GaussianBlur(denoise,(5,5),0)
grey=cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
cv2.imwrite('hulk.png',grey)
original=pytesseract.image_to_string(grey,config=' ')
print(original)
#put 'original' into gtts for creating audio


