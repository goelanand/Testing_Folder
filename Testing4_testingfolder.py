import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = pd.DataFrame({"toothed":["True","True","True","False","True","True","True","True","True","False"],
                     "hair":["True","True","False","True","True","True","False","False","True","False"],
                     "breathes":["True","True","True","True","True","True","False","True","True","True"],
                     "legs":["True","True","False","True","True","True","False","False","True","True"],
                     "species":["Mammal","Mammal","Reptile","Mammal","Mammal","Mammal","Reptile","Reptile","Mammal","Reptile"]},
                    columns=["toothed","hair","breathes","legs","species"])

features = data[["toothed","hair","breathes","legs"]]
target = data["species"]

print(data)


#extending the demo decision tree

dataset=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', names=['animal_name','hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','clas'])
print(dataset.shape)
dataset=dataset.drop('animal_name', axis=1)

train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]

tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
tree2 = DecisionTreeClassifier(criterion = 'gini').fit(train_features,train_targets)


prediction = tree.predict(test_features)
prediction = tree2.predict(test_features)


print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
print("The prediction accuracy is: ",tree2.score(test_features,test_targets)*100,"%")
print('Thanks added')