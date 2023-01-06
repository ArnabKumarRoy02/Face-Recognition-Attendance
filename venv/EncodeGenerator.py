import cv2
import os
import face_recognition
import pickle

# Importing students images
folderPath = 'Face Recognition with Real Time Database/Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentIDs = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIDs.append(os.path.splitext(path)[0])
    # print(path)
    # print(os.path.splitext(path)[0])

print(studentIDs)

def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncoding(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIDs]
print("Encoding Complete")


file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")