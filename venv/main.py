import cv2
import numpy as np
import os
import pickle
import face_recognition
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("Face Recognition with Real Time Database/Resources/background.png")

# Importing the mode images into the list
folderModePath = 'Face Recognition with Real Time Database/Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('Face Recognition with Real Time Database/venv/EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded Successfully")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    imgBackground[162:162+480, 55:55+640] = img
    
    # To show that webcam is active 
    imgBackground[44:44+633, 808:808+414] = imgModeList[0]
    # To show students record details
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]
    # To show that the student's attendance is marked
    imgBackground[44:44+633, 808:808+414] = imgModeList[2]
    # To show that the student's attendance is already marked
    imgBackground[44:44+633, 808:808+414] = imgModeList[3]

    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("faceDistance", faceDistance)

        matchIndex = np.argmin(faceDistance)
        # Print the index of the user trained data
        # Suppose user trained data is in index 1 of the total array
        # So the output would be Match Index 1
        # print("Match Index", matchIndex)

        # If a face is matched with the any index, it will print
        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)