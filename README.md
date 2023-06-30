# Handwritten-digit-recognition2


pip install pyscreenshot 





def one_time():
    import pyscreenshot as ImageGrab
    import time
    images_folder = "captured_images/9/"
    for i in range(0,5):
        time.sleep(8)
        im=ImageGrab.grab(bbox=(60,170,400,550))
        print("saved..",i)
        im.save(images_folder+str(i)+'.png')
        print("clear screen now and redraw now..")





import cv2
import csv
import glob

header =["label"]
for i in range(0,784):
    header.append("pixel"+str(i))
with open('dataset.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for label in range(10):
        dirList = glob.glob("captured_images/"+str(label)+"/*.png")
        
        for img_path in dirList:
            im= cv2.imread(img_path)
            im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray,(15,15), 0)
            roi= cv2.resize(im_gray,(28,28), interpolation=cv2.INTER_AREA)
            
            data=[]
            data.append(label)
            rows, cols = roi.shape
            
            for i in range(rows):
                for j in range(cols):
                    k =roi[i,j]
                    if k>100:
                        k=1
                    else:
                        k=0
                    data.append(k)
            with open('dataset.csv','a') as f:
                writer = csv.writer(f)
                writer.writerow(data) 





import pandas as pd
from sklearn.utils import shuffle
data =pd.read_csv('dataset.csv')
data=shuffle(data)
data





X = data.drop(["label"],axis=1)
Y = data["label"]





get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import cv2
idx = 314
img = X.loc[idx].values.reshape(28,28)
print(Y[idx])
plt.imshow(img)




from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y, test_size = 0.2)




import joblib
from sklearn.svm import SVC
classifier=SVC(kernel="linear", random_state=6)
classifier.fit(train_x,train_y)
joblib.dump(classifier, "model/digit_recognizer")





from sklearn import metrics
prediction=classifier.predict(test_x)
print("Accuracy= ",metrics.accuracy_score(prediction,test_y))





import joblib
import cv2
import numpy as np 
import time
import pyscreenshot as ImageGrab

model=joblib.load("model/digit_recognizer")
images_folder="img/"

while True:
    img=ImageGrab.grab(bbox=(60,170,400,500))
    
    img.save(images_folder+"img.png")
    im = cv2.imread(images_folder+"img.png")
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15,15), 0)
    
    ret, im_th = cv2.threshold(im_gray,100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_th, (28,28), interpolation =cv2.INTER_AREA)
    
    rows,cols=roi.shape
    
    X = []
    
    for i in range(rows):
        for j in range(cols):
            k = roi[i,j]
            if k>100:
                k=1
            else:
                k=0
            X.append(k)
            
    predictions =model.predict([X])
    print("prediction:",predictions[0])
    cv2.putText(im, "prediction is : "+str(predictions[0]), (20,20), 0, 0.8,(0,255,0),2,cv2.LINE_AA)
    
    cv2.startWindowThread()
    cv2.namedWindow("Result")
    cv2.imshow("Result",im)
    cv2.waitKey(10000)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()


