import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_faces(zipfile_path):
    faces = {}
    with zipfile.ZipFile(zipfile_path) as facezip:
        for filename in facezip.namelist():
            if not filename.endswith(".jpg"):
                continue
            with facezip.open(filename) as image:

                faces[filename] = cv2.imdecode(np.frombuffer(
                    image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return faces


def show_sample_faces(faces):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
    faceimages = list(faces.values())[-16:]
    for i in range(16):
        axes[i % 4][i//4].imshow(faceimages[i], cmap="gray")
    print("Showing sample faces")
    plt.show()
