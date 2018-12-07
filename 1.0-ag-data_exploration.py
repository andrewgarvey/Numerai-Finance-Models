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

# Basic stuff  

train.describe().transpose()

train.isna().sum() 










