import face_recognition
import cv2
import numpy as np
from FE_branch import *

import uuid
import dlib
from classification import Inference

# from firebase_admin import credentials, firestore
import time
import requests
import threading


def face_plus_plus(img, name, new):
    try:
        global fe
        global inf

        # Sam call
        if new:
            features = fe.parse_frame(img)

            if type(features) == list:
                features = features[0]

            base = features["attributes"]

            inf_inp = []
            inf_inp.append(base["age"]["value"])
            inf_inp.append(base["gender"]["value"].lower())
            if base["ethnicity"]["value"].lower() == "asian":
                inf_inp.append("asia")
            else:
                inf_inp.append(base["ethnicity"]["value"].lower())

            predict_final = inf.predict(inf_inp)

            print(predict_final)
            requests.get(
                "http://172.20.10.9:8000/getName?name="
                + name
                + "&new="
                + "true"
                + "&age="
                + str(base["age"]["value"])
                + "&gender="
                + base["gender"]["value"]
                + "&ethnicity="
                + base["ethnicity"]["value"]
                + "&beverage="
                + predict_final["drink"]
                + "&food="
                + predict_final["food"]
                + "&side="
                + predict_final["side"]
            )
            stri = str(base)
            print("sent")
            cv2.putText(img, base, (6, 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        else:
            requests.get(
                "http://172.20.10.9:8000/getName?name="
                + name
                + "&new="
                + "false"
                + "&age="
                + "null"
                + "&gender="
                + "null"
                + "&ethnicity="
                + "null"
            )

    except Exception as e:
        print(e)


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)
# video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# video_capture.set(cv2.CAP_PROP_EXPOSURE, 0.01)

# Load a sample picture and learn how to recognize it.
max_image = face_recognition.load_image_file("max.jpg")
max_face_encoding = face_recognition.face_encodings(max_image)[0]

# Load a second sample picture and learn how to recognize it.
# nikhil_image = face_recognition.load_image_file("nikhil.jpg")
# nikhil_face_encoding = face_recognition.face_encodings(nikhil_image)[0]

joel_image = face_recognition.load_image_file("joel.jpg")
joel_face_encoding = face_recognition.face_encodings(joel_image)[0]

# Load a second sample picture and learn how to recognize it.
sam_image = face_recognition.load_image_file("sam.jpg")
sam_face_encoding = face_recognition.face_encodings(sam_image)[0]

chris_image = face_recognition.load_image_file("chris.jpg")
chris_face_encoding = face_recognition.face_encodings(chris_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    max_face_encoding,
    # nikhil_face_encoding,
    joel_face_encoding,
    sam_face_encoding,
    chris_face_encoding,
]
known_face_names = ["Max. B", "Joel M.", "Sam G.", "Chris P."]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
fe = FeatureExtraction()
inf = Inference()

# Setting up database
# cred = credentials.Certificate("serviceAccountKey.json")
# db = firestore.client()

countt = 0
previous = None
current = None
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            current = name
            if name == "Unknown":
                # logic here
                unique_id = uuid.uuid4()

                name = str(unique_id)
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)
                threading.Thread(
                    target=face_plus_plus, args=(frame, name, True)
                ).start()
            elif current != previous:
                previous = current
                threading.Thread(
                    target=face_plus_plus,
                    args=(frame, name.split(" ")[0].lower(), False),
                ).start()

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

