# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 

 Some basic data cleaning for the model preperation.
@author: 9atg
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
t1 = dt.datetime(1968, 9, 8) # needlessly complicated seed for #learning, never actually do this, dates are a $#!@  https://xkcd.com/927/
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

#TAKES a dict of DF , APPLIES THE SAME CLEANING (basically removing a few rows) AND THEN SPLITS X/Y, RENAMES AND OUTPUTS everything dynamically based on df names

both = dict(test=test,train=train)

def DropSplit(dictionary):
   for i in range(0,len(dictionary)):
       # set var
       name = list(dictionary.keys())[i]
       df = dictionary[name]
       
       # manip var       
       df= df.drop(['id','data_type','era'],axis=1)
       target_index = df.columns.str.contains('target',regex=True) 
       print(df)
       # output var
       #vars()['x_'+name] = df.loc[:,np.invert(target_index)]
       #vars()['y_'+name] = df.loc[:,target_index]
       asdf = 1
       asdf2 = 1
       return asdf 

# use it
DropSplit(both)

#------------------------------------------------------------------------------
# FEATURE REDUCTION
## eg Done via primary component analysis 

### explained variance graph
covar_mat = PCA(len(x_train.columns))
covar_mat.fit(x_train)

variance = covar_mat.explained_variance_ratio_
var = np.cumsum(np.round(variance,3))
var

#plot explained variance
plt.clf
plt.ylabel(' variance explained')
plt.xlabel('# of features')
plt.title('variance explained vs # features')
plt.plot(var)
plt.savefig('variance vs # features.png')


## RFE/PCA/LDA isn't going to work well given our constraints

#FEATURE SELECTION ? 

# ADDING VALUE, pretty sure basics aren't gonna cut it when dealing with actual competitions, 


# Dimensionality increase via tSNE? 





'''
NOTES 
-  NN, RF, or some combo modeling (bagging, boosting)  gonna give best results


'''