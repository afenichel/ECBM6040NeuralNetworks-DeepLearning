import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.nnet.neighbours import images2neibs

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def plot_mul(c, D, im_num, X_mn, num_coeffs):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the images
        n represents the maximum dimension of the PCA space.
        m represents the number of images

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean image
    '''
    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, im_num]
            Dij = D[:, :nc]
            plot(cij, Dij, X_mn, axarr[i, j])

    f.savefig('/home/ubuntu/output/hw1b_im{0}.png'.format(im_num))
    plt.close(f)
    os.system('sudo cp -r /home/ubuntu/output /home/afenichel14 ')
    # scp ~afenichel14/Documents/Columbia/Neural\ Networks\ and\ Deep\ Learning/e6040_hw1_alf2178/hw1a.py afenichel14@dev-ml-datalearning-01.use1.huffpo.net:.
    # scp -r afenichel14@dev-ml-datalearning-01.use1.huffpo.net:./output /Users/afenichel14/Documents/Columbia/Neural\ Networks\ and\ Deep\ Learning/e6040_hw1_alf2178


def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in the image)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of a image

    imname: string
        name of file where image will be saved.
    '''
    f,ax=plt.subplots(1,1)
    Dij=D[:,0:16]
    recon_im=np.zeros((4*sz,4*sz))
    idx=0
    for j in range(0,4*sz,sz):
        for i in range(0,4*sz,sz):
            recon_im[i:i+sz,j:j+sz]=Dij[:,idx].reshape(sz,sz).T
            idx+=1

    ax.axis('off')
    ax.imshow(recon_im.T, cmap = plt.get_cmap('gray'))
    f.savefig('/home/ubuntu/%s' %imname)
    plt.close(f)
    os.system('sudo cp -r /home/ubuntu/output /home/afenichel14 ') 


def plot(c, D, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and c as
    the coefficient vector
    Parameters
    -------------------
        c: np.ndarray
            a l x 1 vector  representing the coefficients of the image.
            l represents the dimension of the PCA space used for reconstruction

        D: np.ndarray
            an N x l matrix representing first l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in the image)

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to the reconstructed image

        ax: the axis on which the image will be plotted
    '''

    recon_im=np.dot(D,c).reshape(256,256)+X_mn
    
    ax.axis('off')
    ax.imshow(recon_im, cmap = plt.get_cmap('gray'))
    
def main():
    '''
    Read all images(grayscale) from jaffe folder and collapse each image
    to get an numpy array Ims with size (no_images, height*width).
    Make sure to sort the filenames before reading the images
    '''

    path='/home/afenichel14/jaffe/'
    img_list=os.listdir(path)
    img_list.sort()
    im=np.asarray([np.array(Image.open(path+i)) for i in img_list])
    no_images, height, width=im.shape
    im=im.reshape(1,no_images,height,width)
    images=T.tensor4('images')

    neibs=images2neibs(images,neib_shape=(height, width), neib_step=(height, width))
    window_function=theano.function([images], neibs)
    I=window_function(im)


    Ims = I.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''

    N=16
    epsilon=0.1
    r=np.arange(N)
    trainingsteps=200
    rng = np.random
    rng.seed(1)

    x=T.fmatrix("x")
    di=T.vector("di")
    dj=T.fmatrix("dj")
    lamb=T.vector("lamb")
    eta=theano.shared(np.float32(0.1), name="eta")
    
    Xd=T.dot(x,di)
    dd=T.dot(di.T,dj)
    cost=T.dot(-Xd.T,Xd)-T.sum(T.dot(-lamb*dd,dd.T))
    gd=T.grad(cost, di)
    f1=theano.function([x,di,dj,lamb], di-eta*gd)
    f2=theano.function([x,di], T.dot(Xd.T,Xd))
    f3=theano.function([x,dj], T.dot(x,dj).T)
    Lamb=np.zeros(N)
    D=rng.randn(height*width,N)

    for n in range(N):
        Di=D[:,n].reshape((-1,))
        Dj=D[:,r!=n]
        t=0
        diff=1000000
        lamb_new=f2(X,Di.astype(np.float32))
        while (t<trainingsteps) and (diff>epsilon):
            y=f1(X,Di.astype(np.float32),Dj.astype(np.float32),Lamb[r!=n].astype(np.float32))
            Di=y/np.linalg.norm(y)
            lamb_old=lamb_new
            lamb_new=f2(X,Di.astype(np.float32))
            diff=abs(lamb_old-lamb_new)
            t+=1
        D[:,n]=Di    
        Lamb[n]=f2(X,Di.astype(np.float32))

    c = f3(X,D.astype(np.float32))


    for i in range(0,200,10):
        plot_mul(c, D, i, X_mn.reshape((256, 256)),
                 [1, 2, 4, 6, 8, 10, 12, 14, 16])

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')
    os.system('sudo cp -r /home/ubuntu/output /home/afenichel14 ') 



if __name__ == '__main__':    
    main()
