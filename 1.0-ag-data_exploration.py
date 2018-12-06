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


#setup dir
inputdir = 'D:\QUEENS MMAI\823 Finance\Project\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Project\Output'

#-------------------------------------------------------------------------------
# IMPORT FILES

## Read file(s) / directory management
os.chdir(inputdir)
train = pd.read_excel('A2trainData_MMAI.xlsx')
test = pd.read_excel('A2testData_MMAI.xlsx')
os.chdir(outputdir)






















