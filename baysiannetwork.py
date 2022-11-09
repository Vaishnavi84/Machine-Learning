import numpy as np
import csv
import pandas as pd

#read Cleveland Heart Disease data
heartDisease = pd.read_csv('C:\MSC-IT\Machine Learning\Beysian Network\heart.csv')
heartDisease = heartDisease.replace('?',np.nan)

#display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())

#Model Bayesian Network
from pgmpy.models import BayesianNetwork
model=BayesianNetwork([('age','target'),('sex','target'),
                       ('exang','target'),('cp','target'),
                       ('target','restecg'),('target','chol')])
                    
#Learning CPDs using Maximum Likelihood Estimators
from pgmpy.estimators import MaximumLikelihoodEstimator
print('\n Learning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)
                    
# Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)
                    
#computing the Probability of RestEcg given HeartDisease is present
print('\n 1. Probability of HeartDisease given RestEcg')
q=HeartDisease_infer.query(variables=['restecg'],evidence ={'target':1}, joint=False)
print(q['restecg'])
                    
#computing the Probability of HeartDisease given Chestpain
print('\n 2. Probability of HeartDisease given Chestpain')
q=HeartDisease_infer.query(variables=['target'],evidence ={'cp':3}, joint=False)
print(q['target'])
