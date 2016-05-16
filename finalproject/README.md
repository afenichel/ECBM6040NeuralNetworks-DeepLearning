#Final Project

This is the repository for the implementation of Binary Connect  by Francisco Arceo, Michael Bisaha, and Allison Fenichel. 

Contact Information:
    - Allison: alf2178@columbia.edu
    - Michael: mwb2127@columbia.edu
    - Francisco: fja2114@columbia.edu

This repository contains the following:

1. Code Repository (.py files)
    
    Our code is made up of three main python files: train_nn.py, train.py, and train_utils.py. The train_utils file is responsible for pulling in and cleaning each dataset. The train_nn file is used for defining classes for each different type of layer, and defines the train_nn function which runs the epochs, trains the models, and tests the accuracy of the model. The main train file defines the functions for our tests and defines the architecture of our deep networks. It imports train_nn and train_utils in order to pull the data and run it through the layers. The train file holds train_mlp() for training the MNIST data and train_bc() that we use to train SVHN and CIFAR10. 

    Additionally, we leverage the original author's filter_plot function (filter_plot.py file), since it is not at all used in the development of the Binary Connect aglorithm, but simply a small plot useful for formatting the data in a consistent fashion (they process their data before plotting). This was a very specific form of plot, thus, we decided to use their code rather than try to replicate it, which would have needed to be nearly identical in structure in order to match said figures. Most importantly, no other code was leveraged from the author's original work and all other code is our own.

2. Reproducable Jupyter Notebooks (.ipynb files)
    
    Testing of each iteration was done in Jupyter notebooks that document all necessary library imports, function uses, and hyperparameters, as well as building all plots and graphs. The accompanying paper contains several, but not all, of these plots. Separate notebooks were created for our testing implementations of MNIST, SVHN, and CIFAR10.

3. Output folder, holding model results (.csv files)

    This folder holds individual files containting the model testing results for each permutation of parameters we tested. For each of the three datasets we worked with (MNIST, SVHN, CIFAR10) we implemented tests for binary/vs non-binary models, as well as stochastic Binary Connect and deterministic Binary Connect models. In total, we've included results from 18 different testing permutations.
