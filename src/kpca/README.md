# ITBA - Métodos Numéricos Avanzados - Grupo 10
First Advanced Numerical Methods Project: Face Recognition

## Getting Started
These instructions will install the development environment into your local machine.

### Prerequisites
1. Clone the repository
	```
	$ git clone https://github.com/lobo/mna-tp1.git
	```
2. Install Python3 and pip3
	#### MacOS
	A. Install packages
	```
	$ brew install python3
	```
	B. Update the ```PATH``` variable to use the Homebrew's python packages
	```
	$ echo 'export PATH="/usr/local/opt/python/libexec/bin:$PATH" # Use Homebrew python' >> ~/.bash_profile
	$ source ~/.bash_profile
	```  
	#### Ubuntu
	```
	$ sudo apt-get install python3.6 python3-pip
	```

### Build 

On the root directory run the following command.

```
$ pip3 install -r requirements.txt
```
  
## Usage

There are 3 executables on the root directory. 

### PCA data

On the root directory run the following command:

```
$ python3 facespca.py 
```
Doing so will generate PCA data.

### KPCA data

On the root directory run the following command:

```
$ python3 faceskernelpca.py 
```

Doing so will generate PCA data.

### Main program

On the root directory run the following command:

```
$ python3 main.py ./att_faces
```

Doing so will execute the following: 

1. Load images.
2. Substract the mean face.
3. Compute eigenvalues.
4. Compute eigenfaces.
5. Project images.
6. Classify them.
7. Turn camera of your computer on.
8. Track faces it finds.
9. Compare them to "familiar faces". These are the ones located in our "faces database".

  
## Authors
* [Axel Fratoni](https://github.com/axelfratoni)
* [Daniel Lobo](https://github.com/lobo)
* [Fernán Oviedo](https://github.com/foviedoITBA)
* [Gastón Rodríguez](https://github.com/gastonrod)
