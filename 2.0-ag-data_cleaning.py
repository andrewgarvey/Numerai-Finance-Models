# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 

"""

#Debatable if done as group or not
#as different models may excel with different inputs 
#can probably find something that works for everyone,

#import standard packages
import os
import time
import datetime as dt
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import other packages 
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA

#------------------------------------------------------------------------------
# IMPORT FILES
#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Project\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Project\Output'


## Read file(s) / directory management
os.chdir(inputdir)
train = pd.read_csv('numerai_training_data.csv')
test = pd.read_csv('numerai_tournament_data.csv')
os.chdir(outputdir)

# Seed
t1 = dt.datetime(2000,11, 7) # needlessly complicated seed for #learning, never actually do this, dates are a $#!@  https://xkcd.com/927/
t2 = dt.datetime(1992,11, 7) # I just assume all your gifts got lost in the mail

diff = t2-t1
diff_sec = int(diff.total_seconds())
print(diff_sec)

np.random.seed(diff_sec)

#-----------------------------------------------------------------------------
# STORING 

# Store ID/data_type/era of TEST, will want later !!  
test_stored_cols = test.loc[:,['id','data_type','era']]

#------------------------------------------------------------------------------
#CLEANING 
#making this unneccesary function with unneccesary loops for #learning

#APPLIES THE SAME CLEANING (basically removing a few rows) AND THEN SPLITS X/Y, RENAMES AND OUTPUTS everything dynamically based on df names

x_train = [] 

both = dict(test=test,train=train)
for i in range(0,len(both)):
   # set var
   name = list(both.keys())[i]
   df = both[name]
   
   # manip var       
   df= df.drop(['id','data_type','era'],axis=1)
   target_index = df.columns.str.contains('target',regex=True) 
   print(df)
   # output var
   vars()['x_'+name] = df.loc[:,np.invert(target_index)]
   vars()['y_'+name] = df.loc[:,target_index]

#------------------------------------------------------------------------------
# FEATURE REDUCTION
## eg Done via primary component analysis 

### explained variance graph
covar_mat = PCA(len(x_train.columns))
covar_mat.fit(x_train)

variance = covar_mat.explained_variance_ratio_
var = np.cumsum(np.round(variance,3))
print(var)

#can get 99% info with only ~33 variables 
#plot explained variance
plt.clf
plt.ylabel(' variance explained')
plt.xlabel('# of features')
plt.title('variance explained vs # features')
plt.plot(var)
plt.savefig('variance vs # features.png')


## RFE/PCA/ isn't going to work well given our constraints......
