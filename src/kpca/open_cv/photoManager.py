import cv2
import mna.tp01.open_cv.FaceDetection as fd
import os
from PIL import Image
import scipy.misc
from pygame import mixer

def getHighestPhotoNum(files):
    a = -1
    for f in files:
        name = int(f[0])
        if a < name:
            a = name
    return a+1


def saveImg(newImg, personName):
    directory = "../img/" + personName + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = os.listdir(directory)
    photoName = getHighestPhotoNum(files)
    newImg.save(directory + str(photoName) + ".pgm")

mixer.init()
mixer.music.load('../res/camera_sound.mp3')
cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

print(os.path.dirname(os.path.realpath(__file__)))

video_capture = cv2.VideoCapture(0)

while True:

    name = input("Enter your name: ")

    if(name == "q"):
        break

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            # print("The video capture is not working.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(92, 112)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            x, y, w, h = fd.resizeShape(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)


        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('p') and frame is not None and len(faces) > 0:
            newImg = fd.cropImage(frame, fd.resizeFace(faces[0]))
            newImg = fd.resizeImg(newImg)
            saveImg(newImg, name)
            mixer.music.play()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
