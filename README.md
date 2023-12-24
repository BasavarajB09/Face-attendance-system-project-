# Face-attendance-system-project-
import cv2
import face_recognition
import os
from datetime import datetime

# Load images and encode faces
known_faces = []
known_names = []

def load_known_faces():
    for filename in os.listdir('known_faces'):
        image = face_recognition.load_image_file(f'known_faces/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

# Initialize camera
video_capture = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = video_capture.read()

    # Find faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Record attendance
        if name != "Unknown":
            with open('attendance.csv', 'a') as file:
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f'{name},{timestamp}\n')

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video_capture.release()
cv2.destroyAllWindows()

