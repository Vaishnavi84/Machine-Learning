pip install apyori

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from apyori import apriori

data = pd.read_csv("C:\MSC-IT\Machine Learning\Rule Based\order_data.csv",delimiter=" ",header=None)
print(data.head())
print(data.shape)

data_list = []
for row in range(0, 20):
    data_list.append([str(data.values[row,column]) for column in range(0, 9)])
    
rules = apriori(data_list, min_support=0.25, min_confidence=0.6, min_lift=2, min_length=2)
results = list(rules)

print(len(results))

for i in range(0,3):
    print(f"Required Association No. {i+1} is: {results[i]}")
    print('-'*25)
