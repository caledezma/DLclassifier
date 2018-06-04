#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:00:29 2018

@author: carlos
"""

#%%
import nnetFunc
import h5func
import numpy as np


#%% Initialize everything
'''
Train using given data file
'''

huList = [7,10,15,20,25]# Number of hidden units
hlList = [4,5,6] # Number of hidden layers (minus 1)
ol = 3 # Number of units in the output layer
maxIt = 50000 # Number of iterations for learning
 # Stop criteria
JstopDiff = 1e-10
JstopMag = 1e-4
lam = 1# Regularization parameter
lr = 0.005 # learning rate
kval = 10 # Learning iterations

# Message to show before starting training
msg = 'Xval training pECG classification with multiple output: Control (100), Mild ischaemia (010) or Severe ischaemia (001)'

fname = 'trainingData/TPtrainingSetMultiOutputFull.h5'
outname = 'trainingResults/optimalWeightsTPECGmultiOutput' #Filename root to save optimal weights
trainData = h5func.import_h5(fname,['data'])[0][0]



#%% Initialize the forward propagation function
forwardProp = nnetFunc.fProp()

numFeat = trainData.shape[1]-ol
numSamples = trainData.shape[0]

#%% Learn
print(msg)
for hl in hlList:
    for hu in huList: #Train for several values of hidden units
        optimWeights = list([])
        bestF = 0
        bestR = 0
        bestP = 0
        print('Training with ',hu,' hidden units and ',hl+1,' hidden layers.')
        
        evalSet, xValSets = nnetFunc.crossVal(kval,numSamples)
        # Training to use when few pathological cases
        for k in range(kval):
            print('Iteration: ',k+1,'/',kval)
            
            #Save the set to be used for (local) evaluation
            thisEval = xValSets[k]
            
            #Save the set for training
            thisTrain = list([])
            for i in range(len(xValSets)):
                if i != k:        
                    thisTrain +=  xValSets[i]
            thisTrain = np.array(thisTrain)
            
            
            IN = np.random.uniform(-0.12,0.12,(hu,numFeat+1)) # Weights for layer 1
            W = np.random.uniform(-0.12,0.12,(hu,hu+1,hl)) # Weights for hidden layers
            OUT = np.random.uniform(-0.12,0.12,(ol,hu+1)) # Weights for output layer
            costPre = 0
            for i in range(maxIt):
                # Forward propagation and back-propagation gradients
                out,cost,dIN,dW,dOUT = forwardProp(trainData[thisTrain,:-ol],np.atleast_2d(trainData[thisTrain,-ol:]).T,IN,W,OUT,lam)
                IN -= lr * dIN
                W -= lr * dW
                OUT -= lr * dOUT
                if abs(costPre-cost) < JstopDiff or abs(cost) < JstopMag:
            #            print('Stopped because improvement was ', abs(costPre-cost))
                    break
                #    print('Improvement: ', costPre-cost)
                        
                costPre = cost
#                print('Iteration: ', i)
#                print('Cost: ', cost)
#            
            target = trainData[thisEval,-ol:]
            out = forwardProp(trainData[thisEval,:-ol],np.atleast_2d(target).T,IN,W,OUT,lam)[0]
            out[out > 0.5] = 1
            out[out < 0.5] = 0
            F, p, r = nnetFunc.Fscore(out,target.T)
            
            if F >= bestF:
                bestF = F
                bestP = p
                bestR = r
                optimWeights = [IN] + [W] + [OUT] 
            
            print('Precision=',p)
            print('Recall=',r)
            print('F-score=',F,'\n')
            
        target = trainData[evalSet,-ol:]
        out = forwardProp(trainData[evalSet,:-ol],np.atleast_2d(target).T,optimWeights[0],optimWeights[1],optimWeights[2],lam)[0]
        out[out > 0.5] = 1
        out[out < 0.5] = 0
        F, p, r = nnetFunc.Fscore(out,target.T) 
        
        print('Generalization metrics: P=',p,' R=',r,' F-score=',F)
        
        fname = outname + str(hl+1) + 'hl' + str(hu) +'hu.h5'
        vname = ['hidLayers','hidNeurons', 'IN','W','OUT','Fscore','Precision','Recall']
        var = [np.array(hl), np.array(hu), optimWeights[0] , optimWeights[1] , \
               optimWeights[2], np.array(F), np.array(p), np.array(r)]
        attr = [[],[],[],[],[],[],[],[]]
        h5func.export_h5(fname,vname,var,attr)
        
        print('Optimal weights saved to ',fname,'\n')
