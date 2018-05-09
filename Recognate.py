import os
import cv2
import requests
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('Recognizer/TrainningData.yml');
id = 0
RecognatedID = list()
font=cv2.FONT_HERSHEY_SIMPLEX

def makePathImage(id,path):
    for imagePath in os.listdir('DataSet'):   
        if int(imagePath[0]) == id:
            sendImageToServer('DataSet/'+imagePath)
            return
            
def sendfile():
    print('sent')

def sendImageToServer(path):
    url = 'http://192.168.1.35:3333/images'
    files = {'img' : open(path, 'rb')}
    requests.post(url, files=files)
    print('sent')
    return
    
while(True):
    ret,img = cap.read()
    if ret:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id, conf = rec.predict(gray[y:y+h,x:x+w])
            if id not in RecognatedID:
                RecognatedID.append(id)
                makePathImage(id,'DataSet')
            cv2.putText(img,str(id),(x,y+h),font,1.3,(0,0,255))
        cv2.imshow("Face",img)
    if(cv2.waitKey(1) == ord('q')):
        break;
cap.release()
cv2.destroyAllWindows()
