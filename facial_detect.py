import cv2
import dlib
import numpy as np
import time
import random
import json

#constants
FPS = 60
DIM = 16/9
RES = 240
EMO_RES = 240
THRESHOLD = 145
TOLERANCEX = 50
TOLERANCEY = 10
BLINK_TIME = 10
#set up

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(RES*DIM))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(RES))
cv2.namedWindow("Webcam")
emo_screen = np.zeros((int(RES), int(RES*DIM),3), np.uint8)
blinking = BLINK_TIME




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

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, blink, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if right == False:
        try:
            
            
            cnt = max(cnts, key = cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            #cv2.circle(emo_screen, (cx,cy),4, (0, 0, 255), 2)
            #cv2.rectangle(emo_screen,(cx-25+100+320,120),(cx+25+100+320,230), (0,255,0), -1)
            
            if blinking <=3:
                cv2.rectangle(emo_screen,(cx-25+100+320,165),(cx+25+100+320,175), (0,255,0), -1)
            else: 
                cv2.rectangle(emo_screen,(cx-25+100+320,120),(cx+25+100+320,230), (0,255,0), -1)
            
        except:
            pass
    else: 
        try:
            cnt = max(cnts, key = cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
            #cv2.circle(emo_screen, (cx,cy),4, (0, 0, 255), 2)
            
            if blinking <= 3:
                cv2.rectangle(emo_screen,(cx-25-100+320,165),(cx+25-100+320,175), (0,255,0), -1)
            else: 
                cv2.rectangle(emo_screen,(cx-25-100+320,120),(cx+25-100+320,230), (0,255,0), -1)
            
        except:
            pass



   
#find dot that changes least (eg 1)
#let dot be baseline dot for model
#in emotion_register, set corrosponding dot to dot in one of the dicts
#find the difference between the two dots, use difference on all dots
#calculate as normal past that point
def testing(dot):
    cv2.circle(img, (shape[dot][0], shape[dot][1]), 4, (255,0,255), -1 )
    print(dot)

def emotion_register(list, x_base, y_base):
    with open('emotion.json', 'r') as openfile:
        json_emotion = json.load(openfile)
    
    visual = {} 
    for item in json_emotion:#issue is in here
        absolute_change = (abs(x_base - json_emotion[item][0][0]), abs(y_base - json_emotion[item][0][1])) #works as intended: finds the integer amounts needed on x and y to pair up with the face

        for data in json_emotion[item]:
            data[0] += absolute_change[0]
            data[1] += absolute_change[1]
        visual[item] = json_emotion[item]

        visual[item] = (np.absolute(np.mean(np.subtract(list, visual[item]))))
        print(item , "with a score of", visual[item])
    #print(json_emotion)
    final =  min(visual, key=visual.get)
    print("the emotion is:", final)

    for item in json_emotion[final]:
        cv2.circle(img, (item[0], item[1]), 2, (255, 0, 0), -1) #temp 


def blink(time):
    if time >=5:
        time -=1 
        return time
    elif time < 0:
        time = random.randint(BLINK_TIME//2, BLINK_TIME*2)
        return time
    else:
        time -=1 
        return time
        

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

kernel = np.ones((9, 9), np.uint8)

cv2.createTrackbar('dot', "Webcam", 0, 68, testing)
while True:
    timer = time.time()
    
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces: #face detection

        shape = predictor(gray,face)
        rect_to_bb(face)

        #eye detection
        
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold =  THRESHOLD #cv2.getTrackbarPos('threshold', 'image') #change this to int when you have the whole thing done.  Int should be 145
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        emo_screen = np.zeros((720, 1280,3), np.uint8)
        blinking = blink(blinking)
        contouring(thresh[:, 0:mid], mid, img, blinking)
        contouring(thresh[:, mid:], mid, img, blinking, True )

        facial_landmarks = []
        testing(cv2.getTrackbarPos("dot", "Webcam"))
        adjust_x, adjust_y = shape[0][0], shape[0][1]
        for (i, j) in shape[0:68]:
           facial_landmarks.append((i,j))
           cv2.circle(img, (i, j), 2, (0, 255, 0), -1) #temp
        
        emotion_register(facial_landmarks, adjust_x, adjust_y)
        

        
        
       


        
        
        
    cv2.imshow("Emotion Screen", emo_screen)
    #this is for bugfixing and seeing what the screen sees
    cv2.imshow("Webcam", img)

    0xff
    #print("The amount of time, in ms, to reach this frame is: " ,1000*(time.time()-timer)) 
    k = cv2.waitKey(1000//FPS)
    if k == 27: #esc key to end program#
        break
cap.release()