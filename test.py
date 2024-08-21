from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces_Data = []

i = 0

name = input("Enter username: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]  
        resized_image = cv2.resize(crop_image, (50, 50))  
       
        if len(faces_Data) <= 100 and i % 10 == 0:
            faces_Data.append(resized_image)

        i += 1
        cv2.putText(frame, str(len(faces_Data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_Data) == 100:
        break

video.release()
cv2.destroyAllWindows()
faces_Data=np.asarray(faces_Data)
faces_Data= faces_Data.reshape(100, -1)

# names
if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# faces
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_Data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(names, f)