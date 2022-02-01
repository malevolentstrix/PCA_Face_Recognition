import cv2
import sys
from resizeimage import resizeimage
from PIL import Image

def detectFace(imagePath):

    cascPath = "./haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(92, 112)
    )

    return faces

def cropImage(img, face):
    crop_img = img[face[1]:face[3] + face[1], face[0]:face[2] + face[0]]  # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    return crop_img

def resizeImg(img):
    return resizeimage.resize_cover(Image.fromarray(img), [92, 112])

def resizeShapeH(x,y,w,h):
    height = int(w * 92 / 112)
    up = h // 10
    return x, (y - up), w, height


def resizeShape(x,y,w,h):
    width = int(h * 92 / 112)
    right = w // 10
    return (x+right), y, width, h

def resizeFace(face):
    return resizeShape(face[0],face[1],face[2],face[3])





