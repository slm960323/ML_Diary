from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
# import pydotplus

#load training data from dataset
iris = load_iris()

test_idx = [0,50,100] #seperate some data for testing

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx,axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

# dot_data = tree.export_graphviz(clf,
#         out_file=dot_data, 
#         feature_names=iris.feature_names,  
#         class_names=iris.target_names,  
#         filled=True, rounded=True,  
#         impurity=False)

# graph = pydotplus.graph_from_dot_data(dot_data,getvalue())  
# graph.write_pdf("iris.pdf")

with open("iris.dot", 'w') as f:
     f = tree.export_graphviz(clf, out_file=f)

import os
os.unlink('iris.dot')