# Face Recognition using 1DPCA, 2D-PCA and KPCA
Facial recognition is a way of identifying or confirming an individual’s identity using their face. Facial recognition systems can be used to identify people in photos, videos, or in real-time.
The aim of this project is to do a  comparative study on Face Recognition using PCA (Principal Component Analysis), 2D-PCA and KPCA (Kernel Principal Component Analysis)

## PCA IN FACE RECOGNITION
Face recognition has a challenge to perform in real-time. Raw face image may consume a long time to recognize since it suffers from a huge amount of pixels. So, to reduce the number of these facial features, we perform dimensionality reduction or feature extraction, to save time for the decision step. Feature extraction refers to transforming face space into a feature space. In the feature space, the face database is represented by a reduced number of features that retain most of the important information of the original face.

## OBJECTIVE
To do a  comparative study on Face Recognition using PCA (Principal Component Analysis), 2D-PCA and KPCA (Kernel Principal Component Analysis).

## PCA
A dimensionality reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set. 

Given an image of size nxn, and T such images exists.
In 1D PCA, the nxn image matrix is converted to a vector of size n2x1 and then each image is added as each row(or column) in a matrix. This matrix will be of size n2xT. PCA is applied on this matrix.

## 2D PCA
2DPCA  is based on 2D matrices. It has a higher recognition accuracy than PCA.

Given an image of size nxn, and T such images exists.
Each T image matrix of size nxn is tacked to give a matrix of size Txnxn on which PCA is applied.

## KPCA
Kernel Principal Component Analysis is an extension of principal component analysis (PCA) that uses the technique of kernels.

The idea of KPCA relies on the intuition that the data that are not linearly separable in their space, can be made linearly separable by projecting them into a higher dimensional space. The added dimensions are just simple arithmetic operations performed on the original data dimensions.

So the dataset is projected into a higher dimensional feature space. They become linearly separable, and then PCA is applied on this new dataset.

## STEPS IN FACE RECOGNITION

1) Format the Image Matrix
2) Find mean vector and mean- subtracted image matrix
3) Compute Eigenvectora and Eigenvalues from the covariance matrix
4) Choose Eigenfaces
5) Find the weight matrix
6) Recognise the face

In the case of KPCA, first the Kernel matrix is formed, then the kernel function is chosen after which the eiegnvalues, eigenvectors and eigenface is found, and the rest of steps followed is same as above.

## GROUP MEMBERS
| NAME  | ROLL NUMBER |
| ------------- | ------------- |
| GAYATHRI P  | AM.EN.U4AIE20126  |
| JITHIN JOHN  | AM.EN.U4AIE20135 |
| LAKSHMI WARRIER  | AM.EN.U4AIE20143   |
| M DEVIKA  | AM.EN.U4AIE20144  |
| NIVEDITA RAJESH  | AM.EN.U4AIE20153 |

