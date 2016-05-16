"""
Source Code for Homework 4.b of ECBM E6040, Spring 2016, Columbia University

Instructor: Prof. Aurel A. Lazar

"""

import numpy
import copy
import os
import random
import timeit
from sklearn import metrics
from collections import OrderedDict

import theano
from theano import tensor as T

from hw4_utils import load_data, contextwin, shuffle, conlleval, check_dir, shared_dataset
from hw4_nn import myMLP, train_nn
import sys
sys.setrecursionlimit(5000)

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num,nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    return X,Y


def makeY(x):
    y = numpy.zeros(x.shape)
    for i in range(x.shape[1]):
        if(i==0):
            y[:,i] = numpy.mod(x[:,i],2)
        else:
            y[:,i] = numpy.mod(numpy.sum(x[:,0:(i+1)], axis=1),2) 
    return y.astype(int)



#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, cs):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sequence
        x = T.matrix()
        y_sequence = T.ivector('y_sequence')  # labels

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sequence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sequence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sequence_nll = -T.mean(T.log(p_y_given_x_sequence)
                               [T.arange(x.shape[0]), y_sequence])

        sequence_gradients = T.grad(sequence_nll, self.params)

        sequence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sequence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
        self.sequence_train = theano.function(inputs=[x, y_sequence, lr],
                                              outputs=sequence_nll,
                                              updates=sequence_updates,
                                              allow_input_downcast=True)

    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sequence_train(words, labels, learning_rate)
    
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))



#TODO: build and train a MLP to learn parity function
def test_mlp_parity(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             batch_size=100, n_hidden=500, n_hiddenLayers=3,
             verbose=False, nbit=8, activation=T.tanh, nsamples=[1000,500,100]):
    # generate datasets
    numpy.random.seed(100)
    train_set = gen_parity_pair(nbit, nsamples[0])
    valid_set = gen_parity_pair(nbit, nsamples[1])
    test_set  = gen_parity_pair(nbit, nsamples[2])

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    
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
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    
    
    classifier = myMLP(rng=rng, input=x, n_in=nbit, n_hidden=n_hidden, n_out=2, n_hiddenLayers=n_hiddenLayers, activation=activation)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
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

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
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

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    
    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """
    param = {
        'nbits':3,
        'nsamples': [1000, 500, 100],
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 200,
        'nepochs': 60,
        'savemodel': False,
        'folder':'../result'
    }

    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')

    # generate datasets
    
    
    
    
    numpy.random.seed(100)  # Gaurantees consistency across runs
    train_x, train_y = gen_parity_pair(param['nbits'], param['nsamples'][0])
    valid_x, valid_y = gen_parity_pair(param['nbits'], param['nsamples'][1])
    test_x, test_y  = gen_parity_pair(param['nbits'], param['nsamples'][2])
    
    # need to re-create y to match X for training, test, and validation sets
    train_y = makeY(train_x)

    valid_y = makeY(valid_x)

    test_y = makeY(test_x)
    
    # Set nclass to 2 for 0 or 1
    nclasses=2
    vocsize = 2
    nsequences = len(train_x)
    random.seed(100)

    print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=nclasses,
        ne=vocsize,
        cs=param['win']
    )

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    extract_col = train_x.shape[1]-1

    for e in range(param['nepochs']):

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_x, train_y)):
            rnn.train(numpy.asarray(x), numpy.asarray(y), param['win'], param['clr'])
            sys.stdout.flush()

        # evaluation and prediction
        pred_train = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in train_x ])
        pred_valid = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in valid_x ])
        pred_test = numpy.asarray([rnn.classify(numpy.asarray( contextwin(x, param['win'])).astype('int32')) for x in test_x ])

        # F1
        res_train = metrics.f1_score(train_y[:, extract_col], pred_train[:, extract_col])
        res_valid = metrics.f1_score(valid_y[:, extract_col], pred_valid[:, extract_col])
        res_test = metrics.f1_score(test_y[:, extract_col], pred_test[:, extract_col])

        print(numpy.sum(pred_test[:,extract_col]), numpy.sum(train_y[:,extract_col]))


        if res_valid > best_f1:

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid

            if param['verbose']:
                print('NEW BEST: epoch', e,
                      'training F1', res_train,
                      'valid F1', res_valid,
                      'best test F1', res_test)

            param['vf1'], param['tf1'] = res_valid, res_test
            param['be'] = e

        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
           'valid F1', param['vf1'],
           'best test F1', param['tf1'],
           'with the model', param['folder'])

if __name__ == '__main__':
    test_mlp_parity()
