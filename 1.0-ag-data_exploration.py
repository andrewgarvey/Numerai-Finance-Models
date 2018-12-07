# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:39:52 2018

This one is some preleminary stuff about 823 Project Data, specifically the 

@author: 9atg

"""

#import standard packages
import os
import datetime
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import other packages 
import re
from sklearn.metrics import log_loss
import seaborn as sns
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

#------------------------------------------------------------------------------
# I fully expect this to be boring stuff, they don't want us wasting time changing the data, they might even have a rule against it or something haha 

'''
 Lets talk about Numer.ai Scoring, and what it actually means! 

1. Logarithmic Loss based scoring  (must be <0.693)
       a- Validation Logloss: Graded against validation, publically displayed:  
       b- Test Logloss: This is a Hidden holdout set, it lets the know if they actually want to trust our model 
       c- Live Logloss: This is IF they use our stuff, it's determines our payout along with our submitted stake.
       

We seem to have all of them in the dataset, I have no idea which we would use... 
          
Score to benchmark ourselves against is Log loss <0.693: 
    
MATH ASIDE: 
0.693 is recognizable as ln(2), so i did a quick check to see what they are actually asking for... 

-{(y*ln(p) + (1 - y)*ln(1 - p))} is the formula for Log Loss (for binary catorization)

for simplicity in our case it can be reduced to 

= -ln(p)  
where p is probability that you are off by (it says 0 you say 0.5, p =0.5  )

Realistically speaking we just barely need to beat random chance ! 
'''


#------------------------------------------------------------------------------
# 2. Consitency... Required to meet this benchmark in >58% of eras
# Eras, there are many of them , 120 to be exact 
eras = train.loc[:,'era']
print(len(eras.unique()))

#------------------------------------------------------------------------------
#3. You must actually use the same model for everything, 

# you can't just submit several overfitted models 
# that each do well at different eras/person



#------------------------------------------------------------------------------
#THEREFORE....    
#My fully functional model that passes all 3 steps is as follows ...  

#                 0.5*np.ones(len(df),)


basically_a_model = 0.5*np.ones(len(test),)

y_true = eras = train.loc[:,'target_bernie']
step1check = log_loss(y_true,basically_a_model)
print('log_loss-->', round(step1check,3))



#applies to any and all eras >58% of the time
for i in eras.unique():
    index = eras==i
    y_true = train.loc[index,'target_bernie']
    basically_a_model = 0.5*np.ones(len(y_true),)
    print('for ',i,' log_loss-->',round(log_loss(y_true,basically_a_model),3))

# and is ofc the same model each time. 
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Actual Data stuff,

#Boring Dataset is boring
train.describe().transpose()

# 1 ID, useless except they want it submitted with it
# 1 data_type, just tells us train is train and test is validate/test/live, we will probably just use validate
traintype = train.loc[:,'data_type']
print(traintype.unique())
testtype = test.loc[:,'data_type']
print(testtype.unique())
# 50 features, 0 - 1 
# 5 targets, 0 or 1

#-----------------------------------------------------------------------------
#not missing anything 
train.isna().sum() 

# eras actually mildly important due to how scoring works
eras = train.loc[:,'era']
print(len(eras.unique()))

# cat imbalance? NOPE  
col_index = train.columns.str.contains('target',regex=True) 
train.loc[:,col_index].mean()

#test ? still nope
col_index = test.columns.str.contains('target',regex=True) 
test.loc[:,col_index].mean()


#------------------------------------------------------------------------------
#needs more graphs i guess 

#feature50 plot for no real reason
f50 = train.loc[:,'feature50']

plt.clf()
plt.ylabel(' Frequency')
plt.xlabel('Feature 50')
plt.title('Frequency of Feature 50')
plt.hist(f50, bins=50, range= (0,1))
plt.savefig('1.0-ag-feature50.png')

# some sort of correlation matrix also for no real reason... 
plt.clf()
f, ax = plt.subplots(figsize=(15, 15))
corr = train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
plt.savefig('1.0-ag-Correlation Matrix.png')

'''
Model NOTES

- few useless cols, order should be maintained the whole way if they want them back for w/e reason
- CV splits by Era.. worth looking at
- some non PCA based thing probably, although the CAC is great
- a good way to do multi-classification
- NN would be fun i think 
- no imbalance

Presentation Notes
-A graph would be really nice of historical pricing for NMR,
-NMR -> Nuclear Magnetic Resonance (MRI), idk some bad joke

'''
















