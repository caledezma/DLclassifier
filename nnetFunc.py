# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:01:30 2017

Functions used in neural network scripts

@author: cledezma
"""
def listToFloat(lst):
    """
    Function to cast all elements from a list to floats
    """
    for i in range(len(lst)):
        lst[i] = float(lst[i])
    return lst
    
def crossVal(k,N):
    """
    evalSet, xValSets = crossVal(k,N)
    Function to build the sets needed for a k-fold cross-validation
    
    Input: 
    
    k: scalar, saying how many batches need to be built
    
    N: scalar, number of samples in the training dataset
    
    Output:
    
    evalSet: list containing the indeces of 0.25*N randomly taken samples
    from the training data. These should be used for evaluation after k-fold
    cross-validation training is made.
    
    xValSets: list containg k lists, each containing randomly sampled indeces 
    from the 0.75*N samples remaining after evalSet has been constructed.
    
    This function was developed in the Multiscale Cardiovascular Engineering
    Group (MUSE) at University College London by Carlos Ledezma.
    """
    from random import shuffle
    from numpy import arange, ceil, delete, random

    # Initialize all indices
    trainSet = arange(N,dtype=int)
    shuffle(trainSet)
    
    evalSet = trainSet[:int(ceil(N*0.25))]
    trainSet = trainSet[int(ceil(N*0.25)):]
#    
#    #Define the number of indices to be used for evaluation
#    evalNum = int(ceil(N*0.25))
#    evalSet = sample(trainSet,evalNum)
#    
#    trainSet = [i for i in range(N) if i not in evalSet]
#    
    #Take the evaluation indeces away to generate the training indices
#    for i in range(evalNum):
#        del trainSet[trainSet.index(evalSet[i])]
        
    xValTrainNum = int(ceil(len(trainSet)/k))
    xValSets = list([])
#    for i in range(k-1):
#        xValS = sample(trainSet,xValTrainNum)
#        for i in range(xValTrainNum):
#            del trainSet[trainSet.index(xValS[i])]
#            
#        xValSets += [xValS]
    for i in range(k-1):
        train_set_len = len(trainSet)
        xVal_set_idx = random.choice(train_set_len,size=xValTrainNum,replace=False)
        xVal_set = trainSet[xVal_set_idx]
        trainSet = delete(trainSet,xVal_set_idx)
        xValSets += [xVal_set]
        
    xValSets += [trainSet]
    
    return evalSet, xValSets
    
def Fscore(out,target):
    
    """
    F, p, r = Fscore(out,target)     
    
    Determine the precision, recall and F-score of a set of detections.
    
    Input:
    
    out: numpy array containing the output from a given binary classifier (0 or 1)
    
    target: numpy array containing the target classification values (0 or 1)
    corresponding to each entry in out.
    
    Output:
    
    F: float, F-score calculated as F=2*p*r/(p+r)
    
    p: float, precision calculated as p = TP/(TP+FP)
    
    r: float, recall claculated as r = TP/(TP+FN)
    
    In these formulas TP is the sum of all true positives, FP the sum of all
    false positives and FN the sum of all false negatives. 
    
    This function was developed in the Multiscale Cardiovascular Engineering
    Group (MUSE) at University College London by Carlos Ledezma.
    """
    

    myVec1 = out + target
    myVec2 = out - target
    
    TP = myVec1[myVec1==2].size
    FP = myVec2[myVec2==1].size
    FN = myVec2[myVec2==-1].size
    
    if TP+FP != 0:
        p = TP / (TP+FP)
    else:
        p = 0
    
    if TP + FN != 0:  
        r = TP / (TP+FN)
    else:
        r = 0
    
    if p+r != 0:
        F = 2*p*r / (p + r)
    else:
        F = 0
    
    return F,p,r
    

def fProp():
    """
    Returns the Theano-style forward propagation and gradient calculation function
    
    The generalized function will have the following structure:
    
    [o, J, dIN, dW, dOUT] = fProp(X,Y,IN,W,OUT,L)
    
    Inputs:
    
    X: numpy array containing the examples to be forward propagated. Each row
    is an example, each column is a feature.
    
    Y: target values, each item has to correspond to an example in X. If not
    training just pass an array with zeros 
    
    IN: numpy 2D array with weights that map input layer to first hidden layer
    
    W: numpy 3D array with weights that map within hidden layers. W[:,:,i]
    corresponds to the weights mapping from hidden layer i to hidden layer i+1
    
    OUT: numpy 2D array that maps from last hidden layer to output unit
    
    L: regularization parameter
    
    Outputs:
    
    o: output for each example in X
    
    J: cost calculated using negative log-likelihood
    
    dIN, dW, dOUT: partial derivatives of the cost with respect to the weights
    
    This function was developed in the Multiscale Cardiovascular Engineering
    Group (MUSE) at University College London by Carlos Ledezma.
    """    
    
    import theano.tensor as T
    from theano import function
    from theano.gradient import jacobian    
    from theano import scan
    
    # Define the forward propagation function
    # This function will process all examples at the same time
    
    L = T.dscalar('L') # Regularization term
    X = T.dmatrix('X') # Input cases
    numEx = T.shape(X)[0]
    Y = T.dmatrix('Y') # Target
    IN = T.dmatrix('IN') # ANN weights mapping input layer to first hidden layer
    W = T.dtensor3('W') # ANN weigths mapping between hidden layers
    OUT = T.dmatrix('OUT') # ANN weights mapping last hidden layer to output
    
    # Start forward prop by mapping inputs to first hidden layer
    Xb = T.concatenate([T.ones((numEx,1)),X],axis=1) # Add bias term
    a = T.dot(IN,Xb.T) # Linear combination of inputs
    A = T.nnet.relu(a) # ReLU
    
    '''
    Propagate through the network
    
    Each step is as follows:
    
    actb = T.concatenate([T.ones((1,numEx)),act], axis=0) # Add bias term
    b = T.dot(W[:,:,i],actb) # Linear combination of inputs
    B = T.nnet.relu(b) # ReLU
    
    '''
    
    B, update = scan(lambda i, act,W: T.nnet.relu(T.dot(W[:,:,i], T.concatenate([T.ones((1,numEx)),act], axis=0))),\
                            sequences=T.arange(W.shape[2]),\
                            outputs_info=A,\
                            non_sequences=W)
    
    
    B_final = B[-1]
    # Map to output layer
    Bb = T.concatenate([T.ones((1,numEx)),B_final], axis=0) # Add bias term
    o = T.dot(OUT,Bb) # Linear combination of inputs
    o = T.nnet.sigmoid(o) # Sigmoid for classification output
    
    J = T.nnet.nnet.binary_crossentropy(o,Y).sum() / numEx# Calculate cost
    J += L/(2*numEx) * ((W**2).sum().sum().sum() + (OUT**2).sum().sum() + (IN**2).sum().sum())# Add regularization 
    
    # Calculate jacobians of cost
    dIN = jacobian(J,IN)
    dW = jacobian(J,W)
    dOUT = jacobian(J,OUT)
    
    return function([X,Y,IN,W,OUT,L],[o,J,dIN,dW,dOUT])
    #forwardProp = function([X,Y,W1,O,L],[o,J,dW1,dO])
    
def findInputRelevance(W1,W2):
    '''
    This function finds the relevance of each of the input features in a
    fully-connected neural network as secified by Goh(1995). Observe that 
    in the fuly connected network the relative relevance of the features 
    only depends on the weights that map the input layer into the first 
    hidden layer and, in some cases, those that map the first hidden layer
    into the second hidden layer.
    
    Inputs:
        W1: numpy array with the weights that map the input features into the 
        first hidden layer. The shape of W1 must be W1.shape = (hu,numIn), 
        where hu is the number of hidden units in the first layer and numIn
        the number of input features.
        
        W2: numpy array with the weights that map the first hidden layer into the 
        second hidden layer. The shape of W1 must be W1.shape = (hu,hu), 
        where hu is the number of hidden units in the first and second layers.
        
    Outputs:
        I: numpy array with shape (numIn,) containing the relevance of each 
        input feature.
        
    This function was developed in the Multiscale Cardiovascular Engineering
    Group (MUSE) at University College London by Carlos Ledezma.
    '''
    from numpy import zeros
    
    hu = W2.shape[1]-1
    numIn = W1.shape[1] - 1
    numO = W2.shape[0]
    S = zeros((numO,numIn))
    
    for k in range(numO):
        P = zeros((hu,numIn))
        Q = zeros(P.shape)
        Sbuf =  zeros((1,numIn))

        for i in range(1,hu+1):
            for j in range(1,numIn+1):
                P[i-1,j-1] = abs(W2[k,i]) * abs(W1[i-1,j])
                
        for i in range(hu):
            for j in range(numIn):
                Q[i,j] = P[i,j] / P[i,:].sum()
            
            
        for j in range(numIn):
            Sbuf[0,j] = Q[:,j].sum()
            
        for j in range(numIn):
            S[k,j] = Sbuf[0,j] / Sbuf[0,:].sum()
            
    Ibuf = S.sum(axis=0)
    I = zeros(Ibuf.shape)
    
    for i in range(numIn):
        I[i] = Ibuf[i]/Ibuf.sum()
    
    return I