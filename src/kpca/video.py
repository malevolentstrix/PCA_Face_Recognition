import numpy as np
import image
import cv2
import open_cv.FaceDetection as fd

HAAR_CASCADE_FRONTAL_FACE_PATH = "./open_cv/haarcascade_frontalface_default.xml"

FRAME_COLOR = (0, 255, 0)	# Green
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = FRAME_COLOR
TEXT_ELEVATION = 16
TEXT_THICKNESS = 2

def recognize_faces(mean_face, eigen_faces, classifier):
	faceCascade = cv2.CascadeClassifier(HAAR_CASCADE_FRONTAL_FACE_PATH)
	
	frame = cv2.imread('./caprio.jpg')
	# Turn to gray scal
	grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = faceCascade.detectMultiScale(grayscaled_frame, minSize=(image.HORIZONTAL_SIZE, image.VERTICAL_SIZE))
	# Draw a rectangle around detected faces
	for face in faces:
		x, y, width, height = fd.resizeFace(face)
		cv2.rectangle(frame, (x, y), (x + width, y + height), FRAME_COLOR)
		captured_image = fd.cropImage(frame, fd.resizeFace(face))
		captured_image = fd.resizeImg(captured_image)
		captured_image = captured_image.convert('L')	# 'L' stands for grayscale mode
		captured_image = np.array(captured_image).ravel()
		captured_image = (np.array(captured_image) / image.NORMALIZE_FACTOR) - mean_face
		captured_image = np.dot(np.array(captured_image), eigen_faces.transpose())
		name = classifier.predict([captured_image])
		cv2.putText(frame, name[0], (x, y - TEXT_ELEVATION), fontFace=TEXT_FONT, fontScale=1, color=TEXT_COLOR, thickness=TEXT_THICKNESS)
		print("Detected person :", end='')
		print(name[0])