"""
Source Code by Allison Fenichel, Francisco Arceo, Michael Bisaha
ECBM E6040, Spring 2016, Columbia University


This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
[4] https://github.com/MatthieuCourbariaux/BinaryConnect
[5] https://github.com/hantek/BinaryConnect
"""
import numpy 

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
from final_utils import shared_dataset, load_svhn, load_mnist, load_cifar10
from final_nn import myMLP, LeNetConvPoolLayer, HiddenLayer, SVMLayer, LogisticRegression, train_nn

def test_mlp(initial_learning_rate=0.3, final_learning_rate=0.01, 
             L1_reg=0.00, L2_reg=0.000, n_epochs=100,
             batch_size=200, n_hidden=1024, n_hiddenLayers=3,
             verbose=False, stochastic=False, binary=True, 
             which_data='svhn', seedval=12345, outputlayer='Logistic'):
    """
    Wrapper function for training and testing MLP

    :type intial_learning_rate: float
    :param initial_learning_rate: starting learning rate used for the first epoch (factor for the stochastic
    gradient. The learning rate decays at each epoch after this starting value

    :type final_learning_rate: float
    :param final_learning_rate: final learning rate used for the last epoch. The learning rate decays at each
    epoch, sweeping through values from the the starting learning rate to ending learning rate.
    
    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.
    
    :type binary: boolean
    :param binary: defines whether to use Binary Connect (true) or the base version of the network model (false)
    
    :type stochatic: boolean
    :param stochastic: defines whether to use stochastic (true) or deterministic (false) Binary Connect implementation
    
    :type seedval: int
    :param seedval: defines a seed to use for random numer generation
    
    :type outputlayer: string
    :param outputlayer: defines whetehr to use Logistic or SVM (Support Vector Machine) layer as the final output layer of
    the network
    
    :type which_data: string
    :param which_data: indicates which dataset should be trained by the model - 'mnist', 'svhn', or 'cifar10'

    """
    # load the requested dataset
    if which_data not in ('mnist','cifar10','svhn'):
        return 'Need to choose corrrect dataset either "mnist", "svhn", or "cifar10"'

    if which_data=='mnist':
        datasets = load_mnist(outputlayer)
        nins = 28*28*1

    elif which_data=='svhn':
        datasets = load_svhn(outputlayer)
        nins = 32*32*3

    elif which_data=='cifar10':
        datasets = load_cifar10(outputlayer)
        nins = 32*32*3
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    # allocate dfferent variables for svm vs Logistic layers
    if outputlayer=='svm':
        y = T.imatrix('y')
    if outputlayer=='Logistic':
        y = T.ivector('y')
    
    # define learning rate decay rule
    learning_rate_decay = (float(final_learning_rate)/float(initial_learning_rate))**(1./n_epochs)
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})
   
    rng = numpy.random.RandomState(seedval)
    
    ## define MLP classifier with ReLu activation function
    classifier = myMLP(rng=rng, 
                       input=x, 
                       n_in=nins, 
                       n_hidden=n_hidden, 
                       n_out=10, 
                       n_hiddenLayers=n_hiddenLayers, 
                       stochastic=stochastic,
                       binary=binary,
                       activation=T.nnet.relu,
                       outputlayer=outputlayer)
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.cost(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_model_perf = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    
    ## calculate gradients using binarized weights if binary
    if binary:
        W0=theano.shared(classifier.W0, name='W0', borrow=True)    
        updates=[]
        for i in range(classifier.len_params):
            for j in range(n_hiddenLayers+1):
                p=classifier.params[j*(classifier.len_params)+i]
                if p.name in ('beta','gamma','b', 'Wb'):
                    u = p - learning_rate * T.grad(cost, p)
                    updates.append((p, u))
                elif p.name=='W':
                    n_Wb = classifier.params[j*(classifier.len_params)+i+2]
                    u = p - learning_rate * T.grad(cost, n_Wb)
                    u = T.clip(u, -W0, W0)
                    updates.append((p, u))
                else:
                    continue
    elif not binary:
        gparams = T.grad(cost, classifier.params)
        updates = [(p, p - learning_rate * gp) for p, gp in zip(classifier.params, gparams)]

    # compiling a Theano function `train_model` that returns the cost, but
    # at the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(decay_learning_rate, train_model, train_model_perf, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, which_data,
        stochastic, binary, outputlayer, verbose)
    
    # return first hidden layer weights for image plotting
    return classifier.hiddenLayers[0].W.get_value(), classifier.hiddenLayers[0].Wb.get_value()
    
def test_bc(initial_learning_rate=0.1, final_learning_rate=0.01, n_epochs=1000, 
            nkerns=[64, 64, 128, 128, 256, 256], batch_size=200, verbose=False, 
            stochastic=False, binary=True, which_data='svhn', outputlayer='Logistic'):
    """
    Wrapper function for testing a deep Convoluvtional Network

    :type intial_learning_rate: float
    :param initial_learning_rate: starting learning rate used for the first epoch (factor for the stochastic
    gradient. The learning rate decays at each epoch after this starting value

    :type final_learning_rate: float
    :param final_learning_rate: final learning rate used for the last epoch. The learning rate decays at each
    epoch, sweeping through values from the the starting learning rate to ending learning rate.

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.
    
    :type binary: boolean
    :param binary: defines whether to use Binary Connect (true) or the base version of the network model (false)
    
    :type stochatic: boolean
    :param stochastic: defines whether to use stochastic (true) or deterministic (false) Binary Connect implementation
    
    :type outputlayer: string
    :param outputlayer: defines whetehr to use Logistic or SVM (Support Vector Machine) layer as the final output layer of
    the network
    
    :type which_data: string
    :param which_data: indicates which dataset should be trained by the model - 'mnist', 'svhn', or 'cifar10'
    
    
    """

    rng = numpy.random.RandomState(23455)

    if which_data not in ('mnist','cifar10','svhn'):
        return 'Need to choose corrrect dataset either "mnist", "svhn", or "cifar10"'

    # load data set defined in parameters
    if which_data=='mnist':
        datasets = load_mnist(outputlayer)
        nins = 28*28*1
    elif which_data=='svhn':
        datasets = load_svhn(outputlayer)
        nins = 32*32*3
    elif which_data=='cifar10':
        datasets = load_cifar10(outputlayer)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    if outputlayer=='svm':
        y = T.imatrix('y')
    if outputlayer=='Logistic':
        y = T.ivector('y')
                        # [int] labels
    
    # Define function for learning rate decay
    learning_rate_decay = (float(final_learning_rate)/float(initial_learning_rate))**(1./n_epochs)
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/1, 30/1) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 30, 30)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 3, 3),
        poolsize=(1,1),
        stochastic=stochastic,
        binary=binary
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (30-3+1, 30-3+1) = (28, 28)
    # maxpooling reduces this further to (28/2, 28/2) = (14, 14)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 14, 14)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 30,30),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2,2),
        stochastic=stochastic,
        binary=binary
    )
    
    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (14-3+1, 14-3+1) = (12, 12)
    # maxpooling reduces this further to (12/1, 12/1) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 12, 12)
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 14 ,14),
        filter_shape=(nkerns[2], nkerns[1], 3, 3),
        poolsize=(1,1),
        stochastic=stochastic,
        binary=binary
    )
    
    # Construct the fourth convolutional pooling layer
    # filtering reduces the image size to (12-3+1, 12-3+1) = (10, 10)
    # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 5, 5)
    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer2.output,
        image_shape=(batch_size, nkerns[2], 12 ,12),
        filter_shape=(nkerns[3], nkerns[2], 3, 3),
        poolsize=(2,2),
        stochastic=stochastic,
        binary=binary
    )
    
    # Construct the fifth convolutional pooling layer
    # filtering reduces the image size to (5-3+1, 5-3+1) = (3, 3)
    # maxpooling reduces this further to (3/1, 3/1) = (3, 3)
    # 4D output tensor is thus of shape (batch_size, nkerns[3], 3, 3)
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, nkerns[3], 5 ,5),
        filter_shape=(nkerns[4], nkerns[3], 3, 3),
        poolsize=(1,1),
        stochastic=stochastic,
        binary=binary
    )

    # Construct the sixth convolutional pooling layer
    # filtering reduces the image size to (3-3+1, 3-3+1) = (1, 1)
    # maxpooling reduces this further to (1/1, 1/1) = (1, 1)
    # 4D output tensor is thus of shape (batch_size, nkerns[3], 1, 1)
    layer5 = LeNetConvPoolLayer(
        rng,
        input=layer4.output,
        image_shape=(batch_size, nkerns[4], 3, 3),
        filter_shape=(nkerns[5], nkerns[4], 3, 3),
        poolsize=(1,1),
        stochastic=stochastic,
        binary=binary
    )
    # the two HiddenLayers being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[5] * 1 * 1),
    # or (200, 256 * 1 * 1) = (200, 256) with the default values.
    layer6_input = layer5.output.flatten(2)

    layer6 = HiddenLayer(
        rng,
        input=layer6_input,
        n_in=nkerns[5] * 1 * 1,
        n_out=1024,
        activation=T.nnet.relu,
        stochastic=stochastic,
        binary=binary
    )
    
    layer7 = HiddenLayer(
        rng,
        input=layer6.output,
        n_in=1024,
        n_out=1024,
        activation=T.nnet.relu,
        stochastic=stochastic,
        binary=binary
    )
    
    
    # Define output layer based on parameter
    if outputlayer=='Logistic':
            print("Using logistic regression")
            outputRegressionFunction = LogisticRegression

    if outputlayer=='svm':
        print("Using Support Vector Machines")
        outputRegressionFunction = SVMLayer

    layer8 = outputRegressionFunction(
            input=layer7.output,
            n_in=1024,
            n_out=10,
            stochastic=stochastic,
            binary=binary
        )

    # the cost we minimize during training is the NLL of the model
    cost = layer8.cost(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Get training error rate
    train_model_perf = theano.function(
        [index],
        layer8.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    params = layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    total_len_params = len(params)
    

    # Specifying outputs as layed out in BC paper
    if binary:
        W0=theano.shared(layer0.W0, name='W0', borrow=True)
        updates=[]
        for i in range(total_len_params):
            p=params[i]
            if p.name in ('beta', 'gamma', 'b', 'Wb'):
                u = p - learning_rate * T.grad(cost, p)
                updates.append((p, u))
            elif p.name=='W':
                n_Wb = params[i+1]
                u = p - learning_rate * T.grad(cost, n_Wb)
                u = T.clip(u, -W0, W0)
                updates.append((p, u))
            else:
                continue
    
    elif not binary:
        gparams = T.grad(cost, params)
        updates = [(p,p - learning_rate *gp) for p, gp in zip(params, gparams)]
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(decay_learning_rate, train_model,train_model_perf, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, learning_rate, which_data,
        stochastic, binary, outputlayer, verbose)
    
    # return first hidden layer weights for image plotting
    return layer0.W.get_value(), layer0.Wb.get_value()

if __name__ == "__main__":
    test_mlp(initial_learning_rate=0.3, final_learning_rate=0.0001,
             L1_reg=0.000, L2_reg=0.000, n_epochs=1000, batch_size=200,
             n_hidden=1024, n_hiddenLayers=3, verbose=True, 
             stochastic=True, binary=True, which_data='mnist', 
             seedval=12345, outputlayer='svm')