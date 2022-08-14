
import cv2
import dlib
import numpy as np

#constants
FPS = 5
DIM = 16/9
RES = 240
BRIGHTNESS_TOLERANCE = 13

#set up
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(RES*DIM))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RES))


while True:
    _,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4) 
    
    for (x1,y1,w1,h1) in faces: #face detection
        cv2.rectangle(img, (x1,y1), (x1+w1, y1+h1), (0,255,0),2) #finds face and draws a rectangle
        sub_gray = gray[y1:y1+h1//2,x1:x1+w1] # divide by two to only get eyes (eye will always be in top half of face)

        eyes = eye_cascade.detectMultiScale(sub_gray,1.2,4) 
        for (x2,y2,w2,h2) in eyes:#eye detection
            cv2.rectangle(img,(x2+x1,y2+y1),(x2+x1+w2,y2+y1+h2),(255, 0, 0),1) #draws retangles around each eye
            roi= sub_gray[y2: y2+h2,x2:x2+w2]
            roi= cv2.GaussianBlur(roi, (7,7),0) #increases accuracy by blurring anything outside of the drawn rectangle\

            #---below is still not fully implemented, but will be used to detect gaze detection" 
            rows,cols = roi.shape
            _,bw = cv2.threshold(roi,BRIGHTNESS_TOLERANCE,255,cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

            #cv2.imshow("eyes",bw)
    cv2.imshow("Webcam", img)
    0xff
    k = cv2.waitKey(1000//FPS)
    if k == 27: #q key to end program#
        break
cap.release()