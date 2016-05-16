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

def plot_mul(c, D, im_num, X_mn, num_coeffs, n_blocks):
    '''
    Plots nine PCA reconstructions of a particular image using number
    of components specified by num_coeffs

    Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mn: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Iterable
        an iterable with 9 elements representing the number_of coefficients
        to use for reconstruction for each of the 9 plots

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''

    f, axarr = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            nc = num_coeffs[i*3+j]
            cij = c[:nc, n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
            Dij = D[:, :nc]
            plot(cij, Dij, n_blocks, X_mn, axarr[i, j])

        
    f.savefig('/home/ubuntu/output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    os.system('sudo cp -r /home/ubuntu/output /home/afenichel14 ')
    # scp ~afenichel14/Documents/Columbia/Neural\ Networks\ and\ Deep\ Learning/e6040_hw1_alf2178/hw1a.py afenichel14@dev-ml-datalearning-01.use1.huffpo.net:.
    # scp -r afenichel14@dev-ml-datalearning-01.use1.huffpo.net:./output /Users/afenichel14/Documents/Columbia/Neural\ Networks\ and\ Deep\ Learning/e6040_hw1_alf2178

def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

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
    # raise NotImplementedError


def plot(c, D, n_blocks, X_mn, ax):
    '''
    Plots a reconstruction of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mn: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        ax: the axis on which the image will be plotted
    '''
    sz=256/n_blocks
    blocks=np.dot(D,c)
    recon_im=np.zeros((256,256))
    idx=0
    for j in range(0,256,sz):
        for i in range(0,256,sz):
            recon_im[i:i+sz,j:j+sz]=blocks[:,idx].reshape(sz,sz).T+X_mn.T
            idx+=1

    ax.axis('off')
    ax.imshow(recon_im.T, cmap = plt.get_cmap('gray'))
   
    # raise NotImplementedError


def main():
    '''
    Read here all images(grayscale) from jaffe folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''

    path='/home/afenichel14/jaffe/'
    img_list=os.listdir(path)
    img_list.sort()
    im=np.asarray([np.array(Image.open(path+i)) for i in img_list])
    no_images, height, width=im.shape
    im=im.reshape(1,no_images,height,width)


    szs = [16, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
 
        images=T.tensor4('images')
        neibs=images2neibs(images,neib_shape=(sz,sz), neib_step=(sz,sz))
        window_function=theano.function([images], neibs)
        X=window_function(im)

        X_mn = np.mean(X, 0)
        X = X - np.repeat(X_mn.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        w,v=np.linalg.eigh(np.dot(X.T,X))
        eigindex=np.argsort(w)[::-1]
        D=v[:,eigindex]
        c = np.dot(D.T, X.T)

        for i in range(0, 200, 10):
            plot_mul(c, D, i, X_mn.reshape((sz, sz)),
                     num_coeffs=nc, n_blocks=int(256/sz))

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))


if __name__ == '__main__':
    main()
