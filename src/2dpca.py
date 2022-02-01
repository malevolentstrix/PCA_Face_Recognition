from utils import get_faces, get_person_num, show_sample_faces, split_train_test
import numpy as np

faces = get_faces(zipfile_path="./Grp13Dataset.zip")
# show_sample_faces(faces=faces)

faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)
img_height, img_width = faceshape

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

training_set, testing_set = split_train_test(zipfilepath="./Grp13Dataset.zip")


def get_matrix(training_list, img_height, img_width):

    img_mat = np.zeros(
        (len(training_list), img_height, img_width),
        dtype=np.uint8)

    i = 0
    for img in training_list:
        mat = np.asmatrix(training_list[img])
        img_mat[i, :, :] = mat
        i += 1
    print("Matrix Size:", img_mat.shape)
    return img_mat

# def show_eigen_face(mean_subtracted, eig_no, new_bases):
#     ev = new_bases[:, eig_no:eig_no + 1]
#     print(new_bases.shape)
#     print((mean_subtracted[0]@new_bases).shape)
#     print(ev.shape)
#     cv2.imshow("Eigen Face " + str(eig_no),  cv2.resize(np.array((80,50), dtype = np.uint8),(200, 200)))
#     cv2.waitKey()


facematrix = get_matrix(training_set, img_height, img_width)
no_of_images = facematrix.shape[0]

# eqn 5
mean_face = np.mean(facematrix, 0)
# eqn 6
mean_subtracted = facematrix - mean_face


mat_width = facematrix.shape[2]
g_t = np.zeros((mat_width, mat_width))  # mxm

for i in range(no_of_images):

    # multiplying net subtracted image with its transpose and adding in gt
    temp = np.dot(mean_subtracted[i].T, mean_subtracted[i])
    g_t += temp

# dividing by total number of images
g_t /= no_of_images
# gt is the cov matrix - eq 7

# finding eigen values and eigen vectors
eig_val, eig_vec = np.linalg.eig(g_t)

n = 20
eigfaces = eig_vec[:, 0:n]

# finding new coordinates using dot product new bases

weight_matrix = np.dot(facematrix, eigfaces)


def get_best_match(img):
    img_mat = testing_set[img]
    distances = []
    for i in range(no_of_images):
        temp_imgs = weight_matrix[i]
        dist = np.linalg.norm(img_mat@eigfaces - temp_imgs)
        distances += [dist]

    min = np.argmin(distances)
    return(min//8 + 1)


correct_pred = 0
wrong_pred = 0
for img in testing_set:
    person_num, img_num = get_person_num(filename=img)

    best_match = get_best_match(img)
    if person_num == best_match:
        correct_pred += 1
    else:
        wrong_pred += 1
total_pred = correct_pred+wrong_pred

print(f" Correct preds: {correct_pred}/{total_pred}")
print(f" Wrong preds: {wrong_pred}/{total_pred}")

print(f"Accuracy {correct_pred*100/total_pred}%")
