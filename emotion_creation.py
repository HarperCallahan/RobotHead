"""Remake Emotion Creation"""
import dlib
import cv2
import json
from imutils import face_utils
import numpy as np
RES = 240
DIM = 16/9
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(RES*DIM))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RES))
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_68.dat")
saved_points = ()
predictor = dlib.shape_predictor('shape_68.dat')

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left() 
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

while True:
    """Meat of function."""
    _,img = cap.read()
    faces = face_detector(img, 1)
    for face in faces:
        """Makes a bounding box for the face."""
        
        
        (x, y, w, h) = face_utils.rect_to_bb(face)

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
        bound = [x, y]
        shape = predictor(img,face)
        shape = shape_to_np(shape)
        landmarks= []
        for ((x, y)) in shape:
            x = int(x)
            y = int(y)
            """Uses dlib to mark all landmarks - IMPORTANT: Not changing with emotion?"""
            landmarks.append((x - bound[0],y - bound[1]))
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    


    if cv2.waitKey(1) & 0xFF == ord('q'):
        """Emotion register."""
        emotion = input("What is this emotion called? ")
        if emotion == 'null':
            print("no emotion registered, operation cancelled")
            exit()
        w = {}
        with open('emotion.json',"r") as json_file:
            w = json.load(json_file)
            w[emotion] = landmarks
        with open('emotion.json','w') as json_file:
            json.dump(w, json_file)
        
        
        print(f"Data points for {emotion} have been added to emotion.json!")
        break
    cv2.imshow("Webcam", img)
    