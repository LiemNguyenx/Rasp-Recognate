import os
import cv2
import requests
import numpy as np
from PIL import Image


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('Recognizer/TrainningData.yml');
font=cv2.FONT_HERSHEY_SIMPLEX

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('_')[0])
        faces.append(faceNp)
        #print(ID)
        IDs.append(ID)
        # cv2.imshow("trainning",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

def listAllId():
    IDs = []
    for imagePath in os.listdir('DataSet'):   
        IDs.append(int(imagePath[0]));
    return list(set(IDs));

RecognatedIDs = listAllId()
RecognatedIDs.append(0)
DuplicateIDs = list()


path = 'DataSet'


def sendfile():
    print('sent')

def sendImageToServer(path):
    print('start sending ....')
    url = 'http://192.168.1.34:3000/images'
    files = {'img' : open(path,'rb')}
    requests.post(url, files=files)
    os.remove(path)
    print('remove-'+path+'-sent')
    

def checkRecog(id,conf,face):
    path = "DataSend/"+str(id)+"_69.jpg"
    cv2.imwrite(path,face)
    
    if id not in DuplicateIDs:
        print('da nhan ra: '+str(id)+'---'+str(conf))
        # cv2.imshow('unknow',face)
        sendImageToServer(path)
k = 0
notRecognateCount = 0
while(True):
    ret,img = cap.read()
    if ret: 
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.2,3);
       
        for(x,y,w,h) in faces:
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            id, conf = rec.predict(gray[y:y+h,x:x+w])
            if conf < 80:
                
                checkRecog(id,conf,gray[y:y+h,x:x+w])
                DuplicateIDs.append(id)
                print('Da nhan ra:' + str(id))
                # cv2.imshow('not record', gray[y:y+h,x:x+w])
                # cv2.putText(img,str(id),(x,y+h),font,1.2,(0,255,0))
            # cv2.putText(img,str(id),(x,y+h),font,1.2,(0,255,0))
            else:
                notRecognateCount = notRecognateCount + 1
                if(notRecognateCount==5):
                    print('khong nhan ra lan: '+str(notRecognateCount)+'---'+str(conf))
                    cv2.imwrite('DataSet/'+str(max(RecognatedIDs)+1)+'_'+str(k)+'.jpg',gray[y:y+h,x:x+w])
                    print('DataSet/'+str(max(RecognatedIDs)+1)+'_'+str(k)+'.jpg')
                    k = k+1
                    
                    if k == 10:
    
                        I, fa = getImagesWithID('DataSet')
                        rec.train(fa,I)
                        rec.save('Recognizer/TrainningData.yml')
                        rec.read('Recognizer/TrainningData.yml')
                        RecognatedIDs = listAllId()
                        print('traned')
                        k=0
                    
                    cv2.imshow('unknow',gray[y:y+h,x:x+w])
                    notRecognateCount = 0
            cv2.imshow("Face",img)
    if(cv2.waitKey(1) == ord('q')):
        break;
cap.release()
cv2.destroyAllWindows()
