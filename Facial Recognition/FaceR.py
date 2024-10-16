from json import load
import cv2
from cv2 import COLOR_RGB2BGR
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX

donFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/Donald Trump.jpg')
faceLoc=FR.face_locations(donFace)[0]
donFaceEncode=FR.face_encodings(donFace)[0]

nancyFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/Nancy Pelosi.jpg')
faceLoc=FR.face_locations(nancyFace)[0]
nancyFaceEncode=FR.face_encodings(nancyFace)[0]

zainFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/pic.jpg')
faceLoc=FR.face_locations(zainFace)[0]
zainFaceEncode=FR.face_encodings(zainFace)[0]

knownEnodings=[donFaceEncode,nancyFaceEncode,zainFaceEncode]
names=['Donald Trump','Nancy Pelosi','Berserk']

uknownF=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/unknown/u14.jpg')
ukbgr=cv2.cvtColor(uknownF,cv2.COLOR_RGB2BGR)

faceLocs=FR.face_locations(uknownF)
uknownFEncodes=FR.face_encodings(uknownF,faceLocs)

for faceloc,uknownFEncode in zip(faceLocs,uknownFEncodes):
    top,right,bottom,left=faceloc
    print(faceLoc)
    cv2.rectangle(ukbgr,(left,top),(right,bottom),(255,0,0),3)
    name='uknown'
    matches=FR.compare_faces(knownEnodings,uknownFEncode)
    print(matches)
    if True in matches:
        matchIndex=matches.index(True)
        print(matchIndex)
        print(names[matchIndex])
        name=names[matchIndex]
    cv2.putText(ukbgr,name,(left,top),font,0.75,(0,0,255),2)

cv2.imshow('my facws',ukbgr)
cv2.waitKey(5000)