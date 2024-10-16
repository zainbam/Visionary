import os
import pickle 
import cv2
import face_recognition as FR
imgDir='C:\\Users\Dell\Documents\Python\pyAI3.6\demoImages\known'
encodings=[]
names=[]
for root,dirs,files in os.walk(imgDir):
    print('root: ',root)
    print('dir: ',dirs)
    print('files: ',files)
    for file in files:
        print('name: ',file)
        fullFilePath=os.path.join(root,file)
        print(fullFilePath)
        name=os.path.splitext(file)[0]
        print(name)
        mypic=FR.load_image_file(fullFilePath)
        faceLoc=FR.face_locations(mypic)[0]
        faceEncode=FR.face_encodings(mypic)[0]
        encodings.append(faceEncode)
        names.append(name)


with open ('myData.pkl','wb') as f:      
    pickle.dump(encodings,f)
    pickle.dump(names,f)