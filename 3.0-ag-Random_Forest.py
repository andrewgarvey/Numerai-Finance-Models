# -*- coding: utf-8 -*-
"""
Random Forest Numerai Model 

Highly based off of the script from Sakib 

@author: 9atg
"""
# In[1]:


# import standard libraries
import numpy as np
import os as os
import pandas as pd
import time


# In[2]:


# import modelling libraries
from sklearn import linear_model, model_selection
import numerox as nx


# In[3]:


# set the data working directory
inputdir = 'D:\QUEENS MMAI\823 Finance\Project\Input'
outputdir = 'D:\QUEENS MMAI\823 Finance\Project\Output'
os.chdir(inputdir)

# In[4]:


# download the latest numerai dataset
data = nx.download("numerai_dataset.zip")

# to make it faster use an existing dataset
#data = nx.load_zip("numerai_datasets.zip")

os.chdir(outputdir)
# In[5]:


# environment settings
MODEL_NAME = "logistic-regression"
FOLDER_NAME = "submission"


# In[6]:


# extend the logistic model class offered by numerox
class logistic(nx.Model):

    def __init__(self, params):
        self.p = params

    def fit_predict(self, dfit, dpre, tournament):
        model = linear_model.LogisticRegression(C=self.p['C'], 
                                                solver=self.p['solver'], 
                                                multi_class=self.p['multi_class'])
        model.fit(dfit.x, dfit.y[tournament])
        yhat = model.predict_proba(dpre.x)[:, 1]
        return dpre.ids, yhat


# In[7]:


# parameters required for hyper-tuning the model
C = [0.0001, 0.001, 0.01]
solver = ["newton-cg", "lbfgs", "sag", "saga"]
multi_class = ["ovr", "multinomial", "auto"]


# In[8]:


# combination of parameters
parameters = {'C': C,
             'solver': solver,
             'multi_class': multi_class}


# In[9]:


# use grid search cv to find the best parameters
train_data = pd.read_csv(os.path.join(os.getcwd(), "numerai_dataset", "numerai_training_data.csv"), header=0)
X = np.array(train_data.loc[:, :"feature50"])


# In[10]:


# list of tournaments
tournaments = ["bernie", "elizabeth", "jordan", "ken", "charles", "frank", "hillary"]


# In[11]:


# set the directory to save the submissions
os.chdir(os.path.join(os.getcwd(), "..", "modelling", FOLDER_NAME, MODEL_NAME))


# In[ ]:


# define kfold cross validation split
kfold_split = 5

# loop through each tournament and print the input for train and validation
for index in range(0, len(tournaments)):
    # get the tournament name
    tournament = tournaments[index]
    
    print "*********** TOURNAMENT " + tournament + " ***********"
    
    # set the target name for the tournament
    target = "target_" + tournament 
    
    # set the y train with the target variable
    y = train_data.iloc[:, train_data.columns == target].values.reshape(-1,)
    
    # use GroupKFold for splitting the era
    group_kfold = model_selection.GroupKFold(n_splits=kfold_split)
    
    counter = 1
    
    print ">> group eras using kfold split\n"
    
    for train_index, test_index in group_kfold.split(X, y, groups=train_data['era']):
        # X_train takes the 50 features only for training and leave the other columns
        X_train = X[train_index][:,3:]
        # y_train remains the same
        y_train = y[train_index]
        
        print ">> running split #", counter
        
        print ">> finding best params"
        clf = model_selection.GridSearchCV(linear_model.LogisticRegression(), parameters, scoring="neg_log_loss", cv=kfold_split, n_jobs=-1)
        clf.fit(X_train, y_train)
        best_params = clf.best_params_
        print ">> best params: ", best_params

        # create a new logistic regression model for the tournament
        model = logistic(best_params)

        print ">> training info:"
        train = nx.backtest(model, data, tournament, verbosity=1)

        print ">> validation info:"
        validation = nx.production(model, data, tournament, verbosity=1)

        print ">> saving validation info: "
        validation.to_csv(MODEL_NAME + "-" + tournament + "-" + str(counter) + ".csv")
        print ">> done saving validation info"

        print "\n"
        
        counter=counter+1
    

