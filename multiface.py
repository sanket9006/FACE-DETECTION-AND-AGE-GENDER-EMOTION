import cv2
import numpy as np
from model import EMR

# prevents opencl usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# Initialize object of EMR class
network = EMR()
network.build_network()

# In case you want to detect emotions on a video, provide the video file path instead of 0 for VideoCapture.
cap = cv2.VideoCapture('r.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Again find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        # draw box around faces
        for face in faces:
            (x,y,w,h) = face
            frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2) 
            newimg = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
            cv2.imshow('naming',newimg)            
            newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
            result = network.predict(newimg)
            if result is not None:
                maxindex = np.argmax(result[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,EMOTIONS[maxindex],(x,y), font, 1,(255,255,255),2,cv2.LINE_AA) 

    asd=cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    abc=cv2.resize(asd,(900,600))
    cv2.imshow('Video',abc )
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()