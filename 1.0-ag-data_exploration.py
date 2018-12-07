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
       a- Validation Logloss: Graded against validation, publically displayed:  In our data this is "numerai_tournament_data.csv"
       b- Test Logloss: This is a Hidden holdout set, it lets the know if they actually want to trust our model 
       c- Live Logloss: This is IF they use our stuff, it's determines our payout along with our submitted stake.
       
For this project we only care about a. 
          
Score to benchmark ourselves against is Log loss <0.693: 
    
MATH ASIDE: 
0.693 is recognizable as ln(2), so i did a quick check to see what they are actually asking for 

-{(y*log(p) + (1 - y)*log(1 - p))} is the formula for Log Loss (for binary catorization)

at p = 50% and y = 1  

=-{(1*log(0.5)}
= 0.693 

same can be shown for y=0 and p =50% 

Basically this just means you have to be better than Random guess as a benchmark!... but ofc it's not that easy

2. Eras, there are many of them




3. You must actually use the same model for everything, you can't just submit several overfitted models that each do well 







'''
train.describe().transpose()

train.isna().sum() 

eras = train.loc[:,'era']
len(eras.unique())



















