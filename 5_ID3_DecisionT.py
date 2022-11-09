import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas import DataFrame
df_tennis = pd.DataFrame(data=pd.read_csv(r'D:\ML\5_DecisionTID3/PlayTennis.csv'))
print(df_tennis)

def entropy(probs):
    import math
    return sum([-prob*math.log(prob,2)for prob in probs])

def entropy_of_list(a_list):
    from collections import Counter
    cnt=Counter(x for x in a_list) #Counter calculates the proportion of class
    num_instances=len(a_list)*1.0  #=14
    probs=[x/num_instances for x in cnt.values()] #x means number of YES/NO 9/14 and 5/14
    return entropy(probs) #Call Entropy
total_entropy=entropy_of_list(df_tennis['PlayTennis'])
print('\n The total Entropy is:\n',total_entropy)

def information_gain(df,split_attribute_name,target_attribute_name,trace=0):
    df_split=df.groupby(split_attribute_name)
    nobs=len(df.index)*1.0
    df_agg_ent=df_split.agg({target_attribute_name:[entropy_of_list,lambda x:len(x)/nobs]})[target_attribute_name]
    df_agg_ent.columns=['Entropy','PropObservations']
    new_entropy=sum(df_agg_ent['Entropy']*df_agg_ent['PropObservations'])
    old_entropy=entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy


def id3(df,target_attribute_name,attribute_names,default_class=None):
    from collections import Counter
    cnt=Counter(x for x in df[target_attribute_name])# class of YES/NO
    if len(cnt)==1:
	    return next(iter(cnt))#next input data set, or raises StopIteration when EOF is hit
    elif df.empty or (not attribute_names):
	    return default_class #Return None for empty data set
    else:
	    default_class=max(cnt.keys())#Assign the default class as the max between Yes/No
	    gainz=[information_gain(df,attr,target_attribute_name)for attr in attribute_names]
	    index_of_max=gainz.index(max(gainz))#Index of Best_Attribute
	    best_attr=attribute_names[index_of_max]
	    tree={best_attr:{}}#Initiate the tree with the best attribute as a node
	    remaining_attribute_names=[i for i in attribute_names if i!=best_attr]
	    for attr_val,data_subset in df.groupby(best_attr):
		    subtree=id3(data_subset,target_attribute_name,remaining_attribute_names,default_class)
		    tree[best_attr][attr_val]=subtree
	    return tree

attribute_names=list(df_tennis.columns)
attribute_names.remove('PlayTennis')#Remove the class attirbute

#Run Algorithm:
from pprint import pprint
tree=id3(df_tennis,'PlayTennis',attribute_names)
print("\n The resulatnt DECISION TREE is:\n")
print(tree)


#Creation of Training and Test sets and prediction
def classify(instance,tree,default=None):
	attribute=next(iter(tree))
	if instance[attribute] in tree[attribute].keys():
		result=tree[attribute][instance[attribute]]
		if isinstance(result,dict):
			return classify(instance,result)
		else:
			return result


training_data=df_tennis.iloc[1:-4]
print(training_data)

test_data=df_tennis.iloc[-4:]
print(test_data)

train_tree=id3(training_data,'PlayTennis',attribute_names)

test_data['predicted']=test_data.apply(classify,axis=1,args=(train_tree,'yes'))

print(test_data['predicted'])
