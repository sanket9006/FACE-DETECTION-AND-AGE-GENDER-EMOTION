import cv2
import numpy as np
from model import EMR
import os
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
WRN_WEIGHTS_PATH = ".\\pretrained_models\\weights.18-4.06.hdf5"

depth=16
width=8
face_size=64
margin=40
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale=1
thickness=2

cv2.ocl.setUseOpenCL(False)
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

network = EMR()
network.build_network()

model = WideResNet(face_size, depth=depth, k=width)()
model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
fpath = get_file('weights.18-4.06.hdf5',WRN_WEIGHTS_PATH,cache_subdir=model_dir)
model.load_weights(fpath)

face_cascade = cv2.CascadeClassifier(CASE_PATH)
frame=cv2.imread('2.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=10,minSize=(face_size,face_size))


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1, thickness=2):
        x, y = point
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def crop_face(imgarray, section, margin=40, size=64):
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

if faces is not ():
    face_imgs = np.empty((len(faces),face_size,face_size, 3))
    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        face_imgs[i,:,:,:] = face_img
    
    if len(face_imgs) > 0:
        # predicting ages and gender
        results = model.predict(face_imgs)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
    
    # draw results
    for i, face in enumerate(faces):
        label = "{}, {}".format(int(predicted_ages[i]),"F" if predicted_genders[i][0] > 0.5 else "M")
        draw_label(frame, (face[0], face[1]), label)
    
    for face in faces:
        (x,y,w,h) = face
        newimg = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
        result = network.predict(newimg)
        if result is not None:
            maxindex = np.argmax(result[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,EMOTIONS[maxindex],(x,y-30), font, 1,(255,255,255),2,cv2.LINE_AA) 
            
    asd=cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    abc=cv2.resize(asd,(900,600))
    
else:
    print('No faces are present')

cv2.imshow('Keras Faces', abc)
cv2.waitKey(0)
cv2.destroyAllWindows()    
    
    
