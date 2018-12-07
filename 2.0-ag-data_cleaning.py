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
import datetime
import random as rd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import other packages 
from sklearn.metrics import log_loss

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


#-----------------------------------------------------------------------------













'''
NOTES 
- ASK ME MORE FUCKING QUESTIONS!!! I don't like that i generally feel compelled to pick my own model everytime  



'''