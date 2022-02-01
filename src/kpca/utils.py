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


def get_person_num(filename):
    filename = filename.replace("Grp13Person", "").split("/")
    person_num = int(filename[0])
    img_num = int(filename[1].split("_")[0]) % 10
    return (person_num, img_num)


def split_train_test(zipfilepath):
    training_set = {}
    testing_set = {}
    with zipfile.ZipFile(zipfilepath) as facezip:
        for filename in facezip.namelist():
            if not filename.endswith(".jpg"):
                continue

            person_num, img_num = get_person_num(filename=filename)

            # every 0th and 1st pic per person is sent to training set
            if img_num == 0 or img_num == 1:
                with facezip.open(filename) as image:

                    testing_set[filename] = cv2.imdecode(np.frombuffer(
                        image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            else:

                with facezip.open(filename) as image:
                    training_set[filename] = cv2.imdecode(np.frombuffer(
                        image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    return training_set, testing_set

def get_stats(correct_pred,wrong_pred,total_pred):

    Correct_Predic = round(correct_pred/total_pred,3)
    Wrong_Predic =  round(wrong_pred/total_pred,3)
    Accuracy = round((correct_pred*100)/total_pred,3)
    return Correct_Predic, Wrong_Predic, Accuracy