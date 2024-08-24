from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]  
        resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)  
        output = knn.predict(resized_image)
        cv2.rectangle(frame, (x, y), (x+w,y+h),(0,0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w,y+h),(50,50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w,y+h),(50,50, 255), -1)
        cv2.putText(frame, str(output[0]),(x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("frame", imgBackground)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()









# import pickle

# # Load and check names.pkl
# try:
#     with open('data/names.pkl', 'rb') as f:
#         labels = pickle.load(f)
#         print("Contents of names.pkl (Labels):")
#         print(labels)
# except (EOFError, FileNotFoundError) as e:
#     print(f"Error loading names.pkl: {e}")

# # Load and check faces_data.pkl
# try:
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces_data = pickle.load(f)
#         print("\nContents of faces_data.pkl (Faces Data):")
#         print(faces_data)
#         print("\nNumber of face data entries:", len(faces_data))
# except (EOFError, FileNotFoundError) as e:
#     print(f"Error loading faces_data.pkl: {e}")
