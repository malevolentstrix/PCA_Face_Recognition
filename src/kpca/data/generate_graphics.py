import numpy as np
import matplotlib.pyplot as plt

def load_data(f):
    data = []
    for line in f.readlines()[:-1]:
        splitted_line = [float(x) for x in line.strip().split(',')]
        data.append(splitted_line)
    return np.asarray(data)

kpca5data = load_data(open('kpca5.txt'))
kpca9data = load_data(open('kpca9.txt'))
pca5data = load_data(open('pca5.txt'))
pca9data = load_data(open('pca9.txt'))

plt.plot(kpca5data[:,0],kpca5data[:,1])
plt.plot(pca5data[:,0],pca5data[:,1])
plt.title('Azul: KPCA, Naranja: PCA')
plt.xlabel('Cantidad de autocaras')
plt.ylabel('Proporción de aciertos')
plt.savefig('5graphic.png')
plt.show()
plt.plot(kpca9data[:,0],kpca9data[:,1])
plt.plot(pca9data[:,0],pca9data[:,1])
plt.title('Azul: KPCA, Naranja: PCA')
plt.xlabel('Cantidad de autocaras')
plt.ylabel('Proporción de aciertos')
plt.savefig('9graphic.png')
plt.show()
