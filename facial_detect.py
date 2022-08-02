
import cv2
FPS = 10
DIM = 16/9
RES = 96

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(RES*DIM))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RES))
while True:
    _,img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces: 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
    
    cv2.imshow('Webcam', img)
    0xff
    k = cv2.waitKey(1000//FPS)
    if k == 27:
        break
cap.release()