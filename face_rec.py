import face_recognition as fr
import os
import time
import cv2
import face_recognition
import numpy as np
from time import sleep


def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

def classify_face(img,i=0):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    global faces
    global faces_encoded
    global known_face_names

    if i==1:
        img = cv2.imread(img, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "o-OUTSIDER"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            print(name)
            role = name.split('-')[0]
            if(role == 'f'):
                role = 'Faculty'
            elif(role == 's'):
                role = 'Student'
            else:
                role = 'Outsider'

            name = name.split('-')[1]
            print('Role: {}\nName: {}'.format(role,name))

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, 'Role: {}'.format(role), (50,30), font, 1.0, (255, 255, 255), 2)
            cv2.putText(img, 'Name: {}'.format(name), (50,55), font, 1.0, (255, 255, 255), 2)
            


    if i==1:
        while True:
            cv2.imshow('Video', img)
            if((cv2.waitKey(1)) & 0xFF == ord('q')):
                break
    else:
        cv2.imshow('Video', img)
        return

# classify_face("test.jpg",1)


cap = cv2.VideoCapture(0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cv2.resizeWindow('Street Cam', width, height)
ret = True
while ret:
    ret, frame = cap.read()
    classify_face(frame)
    time.sleep(5)
    if((cv2.waitKey(1)) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()