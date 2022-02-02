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
