# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:43:52 2017

Apply forward propagation on data, reading the calculated weights from a
previously saved file

@author: cledezma
"""

import nnetFunc
import h5func
from numpy import atleast_2d
from numpy import hstack

# Read weights
fname = 'trainingResults/optimalWeightsTPECGmultiOutput5h20hu.h5'
vname = ['W','IN','OUT','Precision','Recall']
W, IN, OUT,p,r = h5func.import_h5(fname,vname)[0]

# Read data
fname = 'trainingData/TPtrainingSetMultiOutput.h5'
vname = ['data']
dat = h5func.import_h5(fname,vname)[0][0]

feat = dat[:,:-3]
target = atleast_2d(dat[:,-3:]).T

# Apply forward prop
forwardProp = nnetFunc.fProp()
o = forwardProp(feat,target,IN,W,OUT,0)[0].T

# Compare results with targets
comp = hstack((o,target.T))
