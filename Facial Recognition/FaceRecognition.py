import cv2
from json import load
from cv2 import COLOR_RGB2BGR
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX
print(cv2.__version__)
width=640
height=360
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

donFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/naj.jpg')
faceLoc=FR.face_locations(donFace)[0]
donFaceEncode=FR.face_encodings(donFace)[0]

nancyFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/Nancy Pelosi.jpg')
faceLoc=FR.face_locations(nancyFace)[0]
nancyFaceEncode=FR.face_encodings(nancyFace)[0]

zainFace=FR.load_image_file('C:/Users/Dell/Documents/Python/pyAI3.6/demoImages/known/pic.jpg')
faceLoc=FR.face_locations(zainFace)[0]
zainFaceEncode=FR.face_encodings(zainFace)[0]

knownEnodings=[donFaceEncode,nancyFaceEncode,zainFaceEncode]
names=['Gotia','Nancy Pelosi','Zain Ul Abideen']

while True:
    ignore,  uknownF = cam.read()

   
    ukrgb=cv2.cvtColor(uknownF,cv2.COLOR_RGB2BGR)

    faceLocs=FR.face_locations(ukrgb)
    uknownFEncodes=FR.face_encodings(ukrgb,faceLocs)

    for faceloc,uknownFEncode in zip(faceLocs,uknownFEncodes):
        top,right,bottom,left=faceloc
        print(faceLoc)
        cv2.rectangle(uknownF,(left,top),(right,bottom),(255,0,0),3)
        name='uknown'
        matches=FR.compare_faces(knownEnodings,uknownFEncode)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name=names[matchIndex]
        cv2.putText(uknownF,name,(left,top),font,0.75,(0,0,255),2)
    
    cv2.imshow('my WEBcam', uknownF)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

