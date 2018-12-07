# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 

 Some basic data cleaning for the model preperation.
@author: 9atg
"""

#Debatable if done together or not
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
diff_sec = diff.total_seconds()
print(diff_sec)

np.random.seed(diff_sec)

#-----------------------------------------------------------------------------
# STORING 

# Store ID column and data_type of TEST, will want later !!  
test_id = test.loc[:,'id']
test_data_type = test.loc[:,'data_type']
#------------------------------------------------------------------------------
#CLEANING 
#making this unneccesary function with unneccesary loops for #learning

#TAKES a list of DF , APPLIES THE SAME CLEANING (basically removing a few rows) AND THEN SPLITS X/Y, RENAMES AND OUTPUTS EVERYTHING dynamically
both = [test,train]

def DropSplit(list): 
    for i in list:
        print(i)
        list[i]  = list[i].drop(['id','data_type'],axis=1)
        
        return list [i]

DropSplit(both)


for i in both:
    print(i.columns())


a = print('test','test')

# Drop data_type 


# SPLIT INTO X/Y 

   
    
    
    
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