# from sklearn.neighbors import KNeighborsClassifier

# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# with open('data/names.pkl', 'rb') as f:
#     LABELS = pickle.load(f)

# with open('data/faces_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground = cv2.imread("background.png")

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         crop_image = frame[y:y+h, x:x+w]  
#         resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)  
#         output = knn.predict(resized_image) 
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
#         exist = os.path.isfile("Attendence/Attendence_" + date + ".csv")
#         cv2.rectangle(frame, (x, y), (x+w,y+h),(0,0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w,y+h),(50,50, 255), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w,y+h),(50,50, 255), -1)
#         cv2.putText(frame, str(output[0]),(x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
#         attendence = [str(output [0]), str(timestamp)]
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("frame", imgBackground)
#     k = cv2.waitKey(1)
#     if k == ord('o'):
#         if exist:
#             with open("Attendence/Attendence_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(attendence)
#             csvfile.close()
#         else:
#             with open("Attendence/Attendence_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendence)
#             csvfile.close()
#     if k == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()












import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Ensure necessary directories exist
if not os.path.exists('data/'):
    os.makedirs('data/')
if not os.path.exists('Attendance/'):
    os.makedirs('Attendance/')

# Load existing data if available
if 'names.pkl' in os.listdir('data/') and 'faces_data.pkl' in os.listdir('data/'):
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
else:
    LABELS = []
    FACES = np.empty((0, 2500))  # 50x50 flattened images

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
if len(LABELS) > 0:
    knn.fit(FACES, LABELS)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
COL_NAMES = ['NAME', 'TIME']

def save_new_face(name, resized_image):
    """ Save the new face and update model data """
    global LABELS, FACES, knn
    # Add the new data
    LABELS.extend([name] * 100)  # Add name multiple times to increase weight in training
    new_faces_data = np.repeat(resized_image, 100, axis=0)  # Repeat the image to match the labels
    FACES = np.append(FACES, new_faces_data, axis=0)

    # Save updated data
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(LABELS, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(FACES, f)

    # Retrain the classifier
    knn.fit(FACES, LABELS)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)

        if len(LABELS) > 0:
            output = knn.predict(resized_image)
            recognized_name = output[0]
        else:
            recognized_name = None

        if recognized_name is None or recognized_name not in LABELS:
            name = input("New face detected. Please enter name: ")
            save_new_face(name, resized_image)

        # Draw rectangle and label around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Prepare attendance record
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [recognized_name, timestamp]
        file_path = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(file_path)

        # Save attendance
        with open(file_path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write header if file is new
            writer.writerow(attendance)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
